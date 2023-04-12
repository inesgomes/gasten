import os
import argparse
import numpy as np
from dotenv import load_dotenv
import pandas as pd

import torch
import wandb

from stg.metrics import fid, LossSecondTerm
from stg.datasets import load_dataset
from stg.gan.train import train
from stg.gan.update_g import UpdateGeneratorGAN, UpdateGeneratorGASTEN, UpdateGeneratorGASTEN_MGDA
from stg.metrics.c_output_hist import OutputsHistogram
from stg.utils import load_z, set_seed, setup_reprod, create_checkpoint_path, gen_seed, seed_worker
from stg.utils.plot import plot_metrics
from stg.utils.config import read_config
from stg.utils.checkpoint import checkpoint, construct_gan_from_checkpoint, construct_classifier_from_checkpoint, get_gan_path_at_epoch, load_gan_train_state
from stg.gan import construct_gan, construct_loss
from stg.classifier import ClassifierCache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        required=True, help="Config file")
    return parser.parse_args()


def construct_optimizers(config, G, D):
    g_optim = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(
        config["beta1"], config["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(
        config["beta1"], config["beta2"]))

    return g_optim, d_optim


def train_modified_gan(config, dataset, cp_dir, gan_path, test_noise,
                       fid_metrics, c_out_hist,
                       C, C_name, C_params, C_stats, C_args, weight, fixed_noise, num_classes, device, seed, run_id, s1_epoch):
    print("Running experiment with classifier {} and weight {} ...".format(
        C_name, weight))
    run_name = '{}_{}_{}'.format(C_name, weight, s1_epoch)

    gan_cp_dir = os.path.join(cp_dir, run_name)

    batch_size = config['train']['step-2']['batch-size']
    n_epochs = config['train']['step-2']['epochs']
    n_disc_iters = config['train']['step-2']['disc-iters']
    checkpoint_every = config['train']['step-2']['checkpoint-every']

    G, D, _, _ = construct_gan_from_checkpoint(
        gan_path, device=device)

    g_crit, d_crit = construct_loss(config["model"]["loss"], D)

    g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)

    if num_classes == 2:
        if weight == "mgda":
            g_updater = UpdateGeneratorGASTEN_MGDA(g_crit, C, normalize=False)
        elif weight == "mgda:norm":
            g_updater = UpdateGeneratorGASTEN_MGDA(g_crit, C, normalize=True)
        else:
            g_updater = UpdateGeneratorGASTEN(g_crit, C, alpha=weight)
    else:
        raise NotImplementedError

    early_stop_key = 'conf_dist'
    early_stop_crit = None if 'early-stop' not in config['train']['step-2'] \
        else config['train']['step-2']['early-stop']['criteria']
    early_stop_crit_step_1 = early_stop_crit \
        if 'early-stop' not in config['train']['step-1'] \
        else config['train']['step-1']['early-stop']['criteria']

    early_stop = (early_stop_key, early_stop_crit) \
        if early_stop_crit is not None \
        else (early_stop_key, None)

    set_seed(seed)
    wandb.init(project=config["project"],
               group=config["name"],
               entity="gomes-inesisabel",
               job_type='step-2',
               name=f'{run_id}-{run_name}',
               config={
        'id': run_id,
        'seed': seed,
        'weight': weight,
        'train': config["train"]["step-2"],
        'classifier_loss': C_stats['test_loss'],
        'classifier': C_name,
        'classifier_args': C_args,
        'classifier_params': C_params,
        'step1_epoch': s1_epoch
    })

    _, _, _, eval_metrics = train(
        config, dataset, device, n_epochs, batch_size,
        G, g_optim, g_updater,
        D, d_optim, d_crit,
        test_noise, fid_metrics,
        n_disc_iters,
        early_stop=early_stop,
        start_early_stop_when=('fid', early_stop_crit_step_1),
        checkpoint_dir=gan_cp_dir, fixed_noise=fixed_noise, c_out_hist=c_out_hist,
        checkpoint_every=checkpoint_every,
        classifier=C)

    wandb.finish()

    return eval_metrics


def compute_dataset_fid_stats(dset, get_feature_map_fn, dims, batch_size=64, device='cpu', num_workers=0):
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)

    m, s = fid.calculate_activation_statistics_dataloader(
        dataloader, get_feature_map_fn, dims=dims, device=device)

    return m, s


def main():
    load_dotenv()
    args = parse_args()

    config = read_config(args.config_path)
    print("Loaded experiment configuration from {}".format(args.config_path))

    if "step-1-seeds" not in config:
        run_seeds = [gen_seed() for _ in range(config["num-runs"])]
    else:
        run_seeds = config["step-1-seeds"]

    if "step-2-seeds" not in config:
        step_2_seeds = [gen_seed() for _ in range(config["num-runs"])]
    else:
        step_2_seeds = config["step-2-seeds"]

    device = torch.device(config["device"])
    print("Using device {}".format(device))

    ###
    # Setup
    ###
    pos_class = None
    neg_class = None
    if "binary" in config["dataset"]:
        pos_class = config["dataset"]["binary"]["pos"]
        neg_class = config["dataset"]["binary"]["neg"]

    dataset, num_classes, img_size = load_dataset(
        config["dataset"]["name"], config["data-dir"], pos_class, neg_class)

    num_workers = config["num-workers"]
    print(" > Num workers", num_workers)

    if type(config['fixed-noise']) == str:
        arr = np.load(config['fixed-noise'])
        fixed_noise = torch.Tensor(arr).to(device)
    else:
        fixed_noise = torch.randn(
            config['fixed-noise'], config["model"]["z_dim"], device=device)

    test_noise, test_noise_conf = load_z(config['test-noise'])
    print("Loaded test noise from", config['test-noise'])
    print("\t", test_noise_conf)

    mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    num_runs = config["num-runs"]
    for i in range(num_runs):
        print("##")
        print("# Starting run", i)
        print("##")

        run_id = wandb.util.generate_id()
        cp_dir = create_checkpoint_path(config, run_id)
        with open(os.path.join(cp_dir, 'fixed_noise.npy'), 'wb') as f:
            np.save(f, fixed_noise.cpu().numpy())

        #################
        # Set seed
        seed = run_seeds[i]
        setup_reprod(seed)

        config["model"]["image-size"] = img_size

        G, D = construct_gan(
            config["model"], img_size, device)
        g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)
        g_crit, d_crit = construct_loss(config["model"]["loss"], D)
        g_updater = UpdateGeneratorGAN(g_crit)

        print("Storing generated artifacts in {}".format(cp_dir))
        original_gan_cp_dir = os.path.join(cp_dir, 'step-1')

        ###
        # Step 1 (train GAN with normal GAN loss)
        ###
        if type(config['train']['step-1']) != str:
            batch_size = config['train']['step-1']['batch-size']
            n_epochs = config['train']['step-1']['epochs']
            n_disc_iters = config['train']['step-1']['disc-iters']
            checkpoint_every = config['train']['step-1']['checkpoint-every']

            fid_metrics = {
                'fid': original_fid
            }
            early_stop_key = 'fid'
            early_stop_crit = None \
                if 'early-stop' not in config['train']['step-1'] \
                else config['train']['step-1']['early-stop']['criteria']
            early_stop = None \
                if early_stop_key is None and early_stop_key is None \
                else (early_stop_key, early_stop_crit)

            wandb.init(project=config["project"],
                       group=config["name"],
                       entity="gomes-inesisabel",
                       job_type='step-1',
                       name=f'{run_id}-step-1',
                       config={
                           'id': run_id,
                           'seed': seed,
                           'gan': config["model"],
                           'optim': config["optimizer"],
                           'train': config["train"]["step-1"],
                           'dataset': config["dataset"],
                           'num-workers': config["num-workers"],
                           'test-noise': test_noise_conf,
            })

            step_1_train_state, _, _, _ = train(
                config, dataset, device, n_epochs, batch_size,
                G, g_optim, g_updater,
                D, d_optim, d_crit,
                test_noise, fid_metrics,
                n_disc_iters,
                early_stop=early_stop,
                checkpoint_dir=original_gan_cp_dir, fixed_noise=fixed_noise,
                checkpoint_every=checkpoint_every)

            wandb.finish()
        else:
            original_gan_cp_dir = config['train']['step-1']
            step_1_train_state = load_gan_train_state(original_gan_cp_dir)

        print(" > Start step 2 (gan with modified loss)")

        if 'step-1-epochs' not in config['train']['step-2']:
            step_1_epochs = ["best"]
        else:
            step_1_epochs = config['train']['step-2']['step-1-epochs']
        step_1_epochs = list(set(step_1_epochs))

        classifier_paths = config['train']['step-2']['classifier']
        weights = config['train']['step-2']['weight']

        ###
        # Train modified GAN
        ###
        mod_gan_seed = step_2_seeds[i]

        for c_path in classifier_paths:
            C_name = os.path.splitext(os.path.basename(c_path))[0]
            C, C_params, C_stats, C_args = construct_classifier_from_checkpoint(
                c_path, device=device)
            C.to(device)
            C.eval()
            C.output_feature_maps = True

            class_cache = ClassifierCache(C)

            def get_feature_map_fn(images, batch_idx, batch_size):
                return class_cache.get(images, batch_idx, batch_size, output_feature_maps=True)[1]

            dims = get_feature_map_fn(
                dataset.data[0:1].to(device), 0, 1).size(1)

            print(" > Computing statistics using original dataset")
            mu, sigma = compute_dataset_fid_stats(
                dataset, get_feature_map_fn, dims, device=device, num_workers=num_workers)
            print("   ... done")

            our_class_fid = fid.FID(get_feature_map_fn, dims,
                                    test_noise.size(0), mu, sigma, device=device)

            conf_dist = LossSecondTerm(class_cache)

            fid_metrics = {
                'fid': original_fid,
                'focd': our_class_fid,
                'conf_dist': conf_dist,
            }

            c_out_hist = OutputsHistogram(class_cache, test_noise.size(0))

            step2_metrics = []

            for s1_epoch in step_1_epochs:
                if s1_epoch == "best":
                    epoch = step_1_train_state['best_epoch']
                elif s1_epoch == "last":
                    epoch = step_1_train_state['epoch']
                else:
                    epoch = s1_epoch

                gan_path = get_gan_path_at_epoch(
                    original_gan_cp_dir, epoch=epoch)
                if not os.path.exists(gan_path):
                    print(
                        f" WARNING: gan at epoch {epoch} not found. skipping ...")

                for weight in weights:
                    eval_metrics = train_modified_gan(config, dataset, cp_dir, gan_path,
                                                      test_noise, fid_metrics, c_out_hist,
                                                      C, C_name, C_params, C_stats, C_args,
                                                      weight, fixed_noise, num_classes, device, mod_gan_seed, run_id, epoch)

                    step2_metrics.append(pd.DataFrame(
                        {'fid': eval_metrics.stats['fid'],
                         'conf_dist': eval_metrics.stats['conf_dist'],
                         's1_epochs': [epoch]*len(eval_metrics.stats['fid']),
                         'weight': [weight]*len(eval_metrics.stats['fid']),
                         'epoch': [i+1 for i in range(len(eval_metrics.stats['fid']))]}))

            step2_metrics = pd.concat(step2_metrics)
            plot_metrics(step2_metrics, cp_dir, f'{C_name}-{run_id}')


if __name__ == "__main__":
    main()

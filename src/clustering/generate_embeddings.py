import argparse
from email.charset import add_codec
import os
from datetime import datetime
from dotenv import load_dotenv
from src.utils.config import read_config
from src.clustering.aux import get_gasten_info, get_gan_path
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint
from src.metrics import fid
from src.datasets import load_dataset
import wandb
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt


def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        help="Config file", required=True)
    parser.add_argument("--run_id", dest="run_id",
                        help="Experiment ID (seen in wandb)", required=True)
    parser.add_argument("--epoch", dest="gasten_epoch", required=True)
    parser.add_argument("--acd_threshold", dest="acd_threshold", default=0.1)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    parser.add_argument("--calculate_fid", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()

    # read configs
    config = read_config(args.config)
    device = config["device"]
    batch_size = config['train']['step-2']['batch-size']
    n_images = config['fixed-noise']
    # prepare wandb info
    dataset_id = datetime.now().strftime("%b%dT%H-%M")
    classifier_name, weight, epoch1 = get_gasten_info(config)    

    config_run = {
        'classifier': classifier_name,
        'gasten': {
            'weight': weight,
            'epoch1': epoch1,
            'epoch2': args.gasten_epoch,
        },
        'probabilities': {
            'min': 0.5 - float(args.acd_threshold),
            'max': 0.5 + float(args.acd_threshold)
        }
    }

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type='step-3-amb_img_generation',
                name=dataset_id,
                config=config_run)

    # get GAN
    gan_path = get_gan_path(
        config, args.run_id, config_run['gasten']['epoch2'])
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)

    # get classifier
    C, _, _, _ = construct_classifier_from_checkpoint(
        config['train']['step-2']['classifier'][0], device=device)
    C.eval()

    # remove last layer of classifier to get the embeddings
    C_emb = torch.nn.Sequential(*list(C.children())[0][:-1])
    C_emb.eval()

    # get test set 
    test_set = load_dataset(config["dataset"]["name"], config["data-dir"], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], train=False)[0]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # get the test set embeddings + predictions
    embeddings_tst_array = []
    pred_tst_array = []
    with torch.no_grad():
        for data_tst in test_loader:
            X, _ = data_tst
            embeddings_tst_array.append(C_emb(X.to(device)))
            pred_tst_array.append(C(X.to(device)))
    # concatenate the arrays
    embeddings_tst = torch.cat(embeddings_tst_array, dim=0)
    pred_tst = torch.cat(pred_tst_array, dim=0).cpu().detach().numpy()

    # prepare FID calculation
    if args.calculate_fid:
        mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
        fm_fn, dims = fid.get_inception_feature_map_fn(device)
        fid_metric = fid.FID(fm_fn, dims, n_images, mu, sigma, device=device)

    # create fake images
    test_noise = torch.randn(n_images, config["model"]["z_dim"], device=device)
    noise_loader = DataLoader(TensorDataset(test_noise), batch_size=batch_size, shuffle=False)
    images_array = []
    for idx, batch in enumerate(tqdm(noise_loader, desc='Evaluating fake images')):
        # generate images
        with torch.no_grad():
            netG.eval()
            batch_images = netG(*batch)
        
        # calculate FID score - all images
        if args.calculate_fid:
            max_size = min(idx*batch_size, n_images)
            fid_metric.update(batch_images, (idx*batch_size, max_size))
            # FID for fake images
            wandb.log({"fid_score_all": fid_metric.finalize()})
            fid_metric.reset()

        images_array.append(batch_images)

    # Concatenate batches into a single array
    images = torch.cat(images_array, dim=0)

    # apply classifier to fake images
    with torch.no_grad():
        pred = C(images).cpu().detach().numpy()

    # filter images so that ACD < threshold
    mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
    images_mask = images[mask]
    pred_syn = pred[mask]

    # point to the original positions (needed later for viz)
    # original_pos = np.where(mask.cpu().detach().numpy())[0]

    # count the ambig images
    n_amb_img = images_mask.shape[0]
    wandb.log({"n_ambiguous_images": n_amb_img})

    # calculate FID score in batches - ambiguous images
    if args.calculate_fid:
        image_loader = DataLoader(TensorDataset(images_mask), batch_size=batch_size, shuffle=False)
        for idx, batch in enumerate(tqdm(image_loader, desc='Evaluating ambiguous fake images')):
            max_size = min(idx*batch_size, n_images)
            fid_metric.update(*batch, (idx*batch_size, max_size))
    
        wandb.log({"fid_score_ambiguous": fid_metric.finalize()})
        fid_metric.reset()

    # get the embeddings for the ambiguous images
    with torch.no_grad():
        embeddings_f = C_emb(images_mask)

    # save embeddings and images
    if args.save:
        print("saving data...")
        DIR = f"{os.environ['FILESDIR']}/data/clustering/{dataset_id}"
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        torch.save(C_emb, f"{DIR}/classifier_embeddings.pt")
        thr = float(args.acd_threshold)*10
        torch.save(images_mask, f"{DIR}/images_acd_{thr}.pt")

    # prepare viz
    print("Start visualizing embeddings...")
    alpha = 0.7
    cmap = 'RdYlGn'

    viz_algs = {
        'UMAP': UMAP(n_components=2),
        'TSNE': TSNE(n_components=2),
        'PCA': PCA(n_components=2)
    }

    embeddings_total = torch.cat([embeddings_tst, embeddings_f], dim=0).cpu().detach().numpy()
    size_real = len(embeddings_tst)

    embeddings_tst_cpu = embeddings_tst.cpu().detach().numpy()
    embeddings_f_cpu = embeddings_f.cpu().detach().numpy()

    for name, alg in viz_algs.items():
        red_embs_syn = alg.fit_transform(embeddings_f_cpu)
        plt.scatter(x=red_embs_syn[:, 0], y=red_embs_syn[:, 1], c=pred_syn, cmap=cmap, marker='o', vmin=0, vmax=1)
        wandb.log({f"{name} Embeddings (gen)": wandb.Image(plt)})
        plt.close()

        red_embs_test = alg.fit_transform(embeddings_tst_cpu)
        plt.scatter(x=red_embs_test[:, 0], y=red_embs_test[:, 1], c=pred_tst, cmap=cmap, marker='x', vmin=0, vmax=1)
        wandb.log({f"{name} Embeddings (test set)": wandb.Image(plt)})
        plt.close()

        red_embs_total = alg.fit_transform(embeddings_total)
        real_embs = red_embs_total[:size_real]
        syn_embs = red_embs_total[size_real:]

        plt.scatter(real_embs[:, 0], real_embs[:, 1], c=pred_tst, label='Real Data', cmap=cmap, alpha=alpha, marker='x', vmin=0, vmax=1)
        plt.scatter(syn_embs[:, 0], syn_embs[:, 1], c=pred_syn, label='Synthetic Data', cmap=cmap, alpha=0.5, marker='o', vmin=0, vmax=1)
        plt.legend()
        wandb.log({f"{name} Embeddings (test set + gen)": wandb.Image(plt)})
        plt.close()

    # close wandb
    wandb.finish()
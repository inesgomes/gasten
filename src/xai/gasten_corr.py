import os
import torch
from dotenv import load_dotenv
from src.utils.config import read_config
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint
from captum.attr import DeepLift, GradientShap, Occlusion
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import wandb
import time
import seaborn as sns
import pandas as pd


def get_gasten_info(config):
    classifier_name = config['train']['step-2']['classifier'][0].split(
        '/')[-1].split('.')[0]
    weight = config['train']['step-2']['weight'][0]
    epoch1 = config['train']['step-2']['step-1-epochs'][0]
    return classifier_name, weight, epoch1


def get_gan_path(config, run_id, epoch2):
    project = config['project']
    name = config['name']
    classifier_name, weight, epoch1 = get_gasten_info(config)
    return f"{os.environ['FILESDIR']}/out/{project}/{name}/{run_id}/{classifier_name}_{weight}_{epoch1}/{epoch2}"


if __name__ == "__main__":
    # setup
    load_dotenv()
    config = read_config("experiments/mnist_7v1_1iter.yml")
    device = torch.device(config["device"])
    classifier_name, weight, epoch1 = get_gasten_info(config)
    config_run = {
        'id': "Jul10T16-31_3l4yezei",
        'classifier': classifier_name,
        'gasten': {
            'weight': weight,
            'epoch1': epoch1,
            'epoch2': 40,
        },
        'probabilities': {
            'min': 0.45,
            'max': 0.55,
        }
    }

    # start experiment
    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='gasten-corr',
               name=f"{config_run['id'].split('_')[-1]}_{int(time.time())}",
               config=config_run)

    # get GAN
    gan_path = get_gan_path(
        config, config_run['id'], config_run['gasten']['epoch2'])
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    fixed_noise = torch.randn(
        config['fixed-noise'], config["model"]["z_dim"], device=device)

    # get classifier
    net, _, _, _ = construct_classifier_from_checkpoint(
        config['train']['step-2']['classifier'][0], device=device)

    # create fake images and apply classifier
    with torch.no_grad():
        netG.eval()
        images = netG(fixed_noise)
        net.eval()
        pred = net(images)

    # select the fake images with prob between 0.4 and 0.6
    #mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
    images_sel = images#[mask]
    preds_sel = pred#[mask]

    # create baseline input (for gradient methods)
    reference_input = torch.full(images[0].shape, -1).unsqueeze(0).to(device)

    # methods to test
    methods_args = {
        'gradientshap': {
            'method': GradientShap,
            'args': {
                'inputs': images_sel,
                'baselines': reference_input,
            }
        },
        'deeplift': {
            'method': DeepLift,
            'args': {
                'inputs': images_sel,
                'baselines': reference_input,
            }
        },
        'occlusion': {
            'method': Occlusion,
            'args': {
                'inputs': images_sel,
                'baselines': reference_input,
                'sliding_window_shapes': (1, 2, 2),
            }
        }
    }

    # prepare wandb table
    corr_table = wandb.Table(columns=['method', 'correlation', 'p-value', 'n'])

    df_total = pd.DataFrame()
    for name, method in methods_args.items():
        print(f"Running {name}")
        # compute attributions
        with torch.no_grad():
            interpret = method['method'](net)
            attrs = interpret.attribute(**method['args'])

        # compute ECDFs, diffs and area
        areas = []
        for _, attr in tqdm(enumerate(attrs)):
            attr_flat = np.round(
                attr.cpu().detach().numpy().flatten(), decimals=4)
            pos = attr_flat[attr_flat > 0]
            neg = np.abs(attr_flat[attr_flat < 0])
            n1, bins, _ = plt.hist(pos, 100, density=True, cumulative=-1)
            n2, _, _ = plt.hist(neg, bins, density=True, cumulative=-1)
            areas.append(abs(n1-n2).sum())

        # compute correlation
        corr, pvalue = stats.spearmanr(
            np.array(areas), preds_sel.cpu().detach().numpy())
        corr_table.add_data(name, np.round(corr, decimals=4),
                            np.round(pvalue, decimals=4), len(areas))
        # create dataframe with correlation and areas
        '''
        df = pd.DataFrame({'areas': np.array(areas), 'preds': preds_sel.cpu().detach().numpy()})
        df['method'] = name
        df_total = pd.concat([df_total, df])
        '''
       
        fig = plt.figure()
        plt.scatter(areas, preds_sel.cpu().detach().numpy())
        wandb.log({"viz_correlation": wandb.Image(fig, caption=name)})

    #fig = sns.lmplot(x="areas", y="preds", col="method", data=df_total)
    #wandb.log({"viz_correlation": fig.get_figure()})

    # save table
    wandb.log({"table_correlation": corr_table})
    # close connection
    wandb.finish()

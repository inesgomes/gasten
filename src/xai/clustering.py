import os
import torch
import wandb
from dotenv import load_dotenv
from src.utils.config import read_config
from src.datasets.datasets import get_mnist
from src.datasets import load_dataset
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, DBSCAN, KMeans
from umap import UMAP
import pandas as pd
import seaborn as sns
import time
import argparse
import numpy as np


def get_test_mnist_data(dataset_name, data_dir, batch_size, pos_class=None, neg_class=None):
    # this is the original dataset -> may try to use this initially
    # using test data
    if (pos_class is None) or (neg_class is None):
        dataset = get_mnist(data_dir, train=False)
    else:
        dataset, _, _ = load_dataset(
            dataset_name, data_dir, pos_class, neg_class, False)
    # use data loader to find a batch of images
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    # get first batch of images
    images, _ = next(iter(data_loader))
    return images


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", dest="clustering", default="hdbscan",
                        required=True, help="clustering model")
    parser.add_argument("--r", dest="reduction", default="none",
                         help="reduction model")
    return parser.parse_args()

EXPERIMENT = {
    'none': None,
    'pca': PCA(n_components=2),
    'tsne': TSNE(n_components=2),
    'umap': UMAP(),
    'dbscan': DBSCAN(min_samples=5, eps=0.2),
    'hdbscan': HDBSCAN(min_samples=5),
    'kmeans': KMeans(n_clusters=3, random_state=0, n_init="auto")
}

if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # type of experiment
    reduction = args.reduction
    clustering = args.clustering

    # read configs
    config = read_config("experiments/mnist_7v1_1iter.yml")
    device = torch.device("cpu") #config["device"])
    n_images = config['fixed-noise']
    pos_class = config["dataset"]["binary"]["pos"]
    neg_class = config["dataset"]["binary"]["neg"]

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
            'min': 0.35,
            'max': 0.65,
        }
    }
    
    # start wandb
    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='clustering',
               name=f"gasten_{pos_class}vs{neg_class}-{reduction}-{clustering}_{int(time.time())}",
               config=config_run)
    
    # get data (1 and 7)
    #images = get_test_mnist_data(config["dataset"]["name"], config["data-dir"], config_run['batch_size'], pos_class=pos_class, neg_class=neg_class)
    
    # get GAN
    gan_path = get_gan_path(
        config, config_run['id'], config_run['gasten']['epoch2'])
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    fixed_noise = torch.randn(
        n_images, config["model"]["z_dim"], device=device)

    # get classifier
    net, _, _, _ = construct_classifier_from_checkpoint(
        config['train']['step-2']['classifier'][0], device=device)
    
    # remove last layer of classifier to get the embeddings
    net_emb = torch.nn.Sequential(*list(net.children())[0][:-1])

    # create fake images and apply classifier
    with torch.no_grad():
        netG.eval()
        images = netG(fixed_noise)
        net.eval()
        pred = net(images)
        embeddings = net_emb(images).cpu().detach().numpy()

    # create wandb table from preds
    wandb.log({"probabilities": wandb.Histogram(pred)})

    # remove images with high probability
    mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])

    if reduction == 'none':
        reduced_embeddings = embeddings[mask]
    else:
        # Apply reduction method to the embeddings
        mdl_r = EXPERIMENT[reduction]
        reduced_embeddings = mdl_r.fit_transform(embeddings[mask])

    # apply clustering method to the embeddings
    mdl_c = EXPERIMENT[clustering]
    clustering_result = mdl_c.fit_predict(reduced_embeddings)

    # plot embeddings
    result = pd.DataFrame({'x': reduced_embeddings[:, 0], 'y': reduced_embeddings[:, 1], 'cluster': clustering_result, 'original_pos': np.where(mask)[0]})
    if reduction != 'none':
        fig = sns.scatterplot(x='x', y='y', hue='cluster', palette='viridis', data=result, legend='full')
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.title(f'N={result.shape[0]}')
        wandb.log({f"{reduction}": wandb.Image(fig, caption=f"{reduction} + {clustering} (N={result.shape[0]})")})

    # log the clustering size
    wandb.log({"cluster_sizes": result.groupby('cluster').size().reset_index(name='counts')})

    # save random images in each cluster
    for clu, cluster_examples in result.groupby('cluster'):
        sample_size = min(5, cluster_examples.shape[0])
        # get the original image positions
        mask = np.full(n_images, False)
        mask[cluster_examples.sample(sample_size)['original_pos']] = True
        wandb.log({"cluster_images": wandb.Image(images[mask], caption=f"Cluster {clu}")})

    # close wandb
    wandb.finish()
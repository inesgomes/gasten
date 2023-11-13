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
from sklearn.cluster import HDBSCAN, DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from umap import UMAP
import pandas as pd
from datetime import datetime
import argparse
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pyclustering.cluster.xmeans import xmeans
from scipy.spatial.distance import cdist


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

    # find directory whose name ends with a given id
    for dir in os.listdir(f"{os.environ['FILESDIR']}/out/{config['project']}/{config['name']}"):
        if dir.endswith(run_id):
            return f"{os.environ['FILESDIR']}/out/{project}/{name}/{dir}/{classifier_name}_{weight}_{epoch1}/{epoch2}"

    raise Exception(f"Could not find directory with id {run_id}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        help="Config file", required=True)
    parser.add_argument("--run_id", dest="run_id",
                        help="Experiment ID (seen in wandb)", required=True)
    parser.add_argument("--epoch", dest="gasten_epoch", required=True)
    parser.add_argument("--n_images", dest="n_images",
                        help="Number of images to generate", default=20000)
    parser.add_argument("--acd_threshold", dest="acd_threshold", default=0.1)
    return parser.parse_args()
    

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


def find_closest_point(target_point, dataset):
    #closest_point = None
    min_distance = float('inf')
    closest_position = -1

    for i, data_point in enumerate(dataset):
        distance = euclidean_distance(target_point, data_point)
        if distance < min_distance:
            min_distance = distance
            #closest_point = data_point
            closest_position = i

    return closest_position


def calculate_medoid(cluster_points):
    # Calculate pairwise distances
    distances = cdist(cluster_points, cluster_points, metric='euclidean')

    # Find the index of the point with the smallest sum of distances
    medoid_index = np.argmin(np.sum(distances, axis=0))

    # Retrieve the medoid point
    medoid = cluster_points[medoid_index]

    return medoid


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()

    # read configs
    config = read_config(args.config)
    device = config["device"]
    pos_class = config["dataset"]["binary"]["pos"]
    neg_class = config["dataset"]["binary"]["neg"]
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
            'min': 0.5 - args.acd_threshold,
            'max': 0.5 + args.acd_threshold
        }
    }
    
    # get GAN
    gan_path = get_gan_path(
        config, args.run_id, config_run['gasten']['epoch2'])
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    fixed_noise = torch.randn(
        args.n_images, config["model"]["z_dim"], device=device)

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
        embeddings = net_emb(images)

    # remove images with high probability
    mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
    embeddings_f = embeddings[mask].cpu().detach().numpy()
    original_pos = np.where(mask.cpu().detach().numpy())[0]
    config_run['n_ambiguous_images'] = embeddings_f.shape[0]

    # TODO calculate FID score for the generated images

    # step 1 - reduce dimensionality
    #emb_size = embeddings_f.shape[1]
    #reductions = {
    #    'None': None,
    #    'pca_90': PCA(n_components=0.9),
    #    'pca_70': PCA(n_components=0.7),
    #    'tsne': TSNE(n_components=2),
    #    'umap_half': UMAP(n_components=int(emb_size/2)),
    #    'umap_2third': UMAP(n_components=int(emb_size*2/3)),
    #}
    # step 2 - clustering
    clusterings = {
        'dbscan': DBSCAN(min_samples=5, eps=0.2),
        'hdbscan': HDBSCAN(min_samples=5, store_centers="both"),
        'kmeans': xmeans(embeddings_f, k_max=15),
        'ward': AgglomerativeClustering(distance_threshold=25, n_clusters=None),
        'gaussian_mixture': BayesianGaussianMixture(n_components=3, covariance_type='full', max_iter=1000, random_state=0),
    }

    # one wandb run for each clustering
    for cl_name, cl_method in clusterings.items():
        # start wandb
        wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type=f'clustering_{cl_name}',
               name=f"{dataset_id}_v1",
               config=config_run)
        
        # TODO hyperparameter optimization

        # apply clustering method
        if cl_name == 'kmeans':
            cl_method.process()
            clustering_xmeans = cl_method.get_clusters()
            subcluster_labels = {subcluster_index: i for i, x_cluster in enumerate(clustering_xmeans) for subcluster_index in x_cluster}
            clustering_result = [x_label for _, x_label in sorted(subcluster_labels.items())]
        else:
            # scikit-learn methods
            clustering_result = cl_method.fit_predict(embeddings_f)

        # verify if it worked
        n_clusters = sum(np.unique(clustering_result)>=0)
        if n_clusters > 1:
            # evaluate the clustering
            wandb.log({"silhouette_score": silhouette_score(embeddings_f, clustering_result)})
            wandb.log({"calinski_harabasz_score": calinski_harabasz_score(embeddings_f, clustering_result)})
            wandb.log({"davies_bouldin_score": davies_bouldin_score(embeddings_f, clustering_result)})
            # log cluster information
            wandb.log({"n_clusters": n_clusters})
            cluster_sizes = pd.Series(clustering_result).value_counts().reset_index()
            wandb.log({"cluster_sizes": wandb.Table(dataframe=cluster_sizes)})

            # save images per cluster
            for cl_label, cl_examples in pd.DataFrame({'cluster': clustering_result, 'original_pos': original_pos}).groupby('cluster'):
                # get the original image positions
                if cl_label >= 0:
                    mask = np.full(args.n_images, False)
                    mask[cl_examples['original_pos']] = True
                    wandb.log({"cluster_images": wandb.Image(images[mask], caption=f"{cl_name} - {cl_label} (N = {cl_examples.shape[0]}))")})

            # get prototypes of each cluster
            proto_idx = None
            prototypes = None
            if cl_name == 'hdbscan':
                prototypes = cl_method.medoids_
            elif cl_name == 'dbscan':
                proto_idx = cl_method.core_sample_indices_
            elif cl_name == 'kmeans':
                means = cl_method.get_centers()
                proto_idx = [find_closest_point(mean_point, embeddings_f) for mean_point in means]
            else:
                # calculate the medoid per each cluster whose label is >= 0
                prototypes = [calculate_medoid(embeddings_f[clustering_result == cl_label]) for cl_label in np.unique(clustering_result) if cl_label >= 0]
                
            if (prototypes is not None) & (proto_idx is None):
                 # find centroids in the original data and get the indice
                proto_idx = [np.where(np.all(embeddings_f == el, axis=1))[0][0] for el in prototypes]

            # save images
            if proto_idx is not None:
                # referencing to the original images
                mask = np.full(args.n_images, False)
                mask[original_pos[proto_idx]] = True
                wandb.log({"prototypes": wandb.Image(images[mask], caption=f"{cl_name}")})

        # close wandb
        wandb.finish()

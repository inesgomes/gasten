import argparse
import numpy as np
import os
import pandas as pd
import wandb
from dotenv import load_dotenv


from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, DBSCAN, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from umap import UMAP

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pyclustering.cluster.xmeans import xmeans


import matplotlib.pyplot as plt

from src.utils.config import read_config


from src.clustering.aux import get_gasten_info, get_gan_path, calculate_medoid, find_closest_point


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
    return parser.parse_args()


def create_cluster_image():
    """_summary_
    """
    ALL_VS_ALL = False

    # setup
    # TODO get embeddings
    # TODO get dataset_id

    n_images = config['fixed-noise']
    emb_size = embeddings_f.shape[1]    

    if ALL_VS_ALL:
        # step 1 - reduce dimensionality
        reductions = {
            'None': None,
        #    'pca_90': PCA(n_components=0.9),
        #    'pca_70': PCA(n_components=0.7),
        #    'tsne': TSNE(n_components=2),
            'umap_sml': UMAP(n_components=15),
            'umap_half': UMAP(n_components=int(emb_size/2)),
        #    'umap_2third': UMAP(n_components=int(emb_size*2/3)),
        }
        # step 2 - clustering
        clusterings = {
            'dbscan': DBSCAN(min_samples=5, eps=0.2),
            'hdbscan': HDBSCAN(min_samples=5, store_centers="both"),
            'kmeans': None,
            'ward': AgglomerativeClustering(distance_threshold=25, n_clusters=None),
            'gaussian_mixture': BayesianGaussianMixture(n_components=3, covariance_type='full', max_iter=1000, random_state=0),
        }
    else:
        # step 1 - reduce dimensionality
        reductions = {
            'umap_sml': UMAP(n_components=15),
        }
        # step 2 - clustering
        clusterings = {
            'dbscan': DBSCAN(min_samples=5, eps=0.2)
        }

    # one wandb run for each clustering
    for cl_name, cl_method in clusterings.items():
        for red_name, red_method in reductions.items():
            # start wandb
            config_run['clustering_method'] = cl_name
            config_run['reduce_method'] = red_name

            job_name = f"{cl_name}_{red_name}" if red_name != "None" else cl_name
            wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-4-clustering_{job_name}',
                name=f"{dataset_id}_v2",
                config=config_run)
            
            # apply reduction method
            embeddings_red = red_method.fit_transform(embeddings_f) if red_name != "None" else embeddings_f

            # if string includes 'kmeans' then apply xmeans
            if cl_name == 'kmeans' :
                # define here the instance
                cl_method = xmeans(embeddings_red, k_max=15)
                cl_method.process()
                clustering_xmeans = cl_method.get_clusters()
                subcluster_labels = {subcluster_index: i for i, x_cluster in enumerate(clustering_xmeans) for subcluster_index in x_cluster}
                clustering_result = [x_label for _, x_label in sorted(subcluster_labels.items())]
            else:
                # scikit-learn methods
                clustering_result = cl_method.fit_predict(embeddings_red)

            # verify if it worked
            n_clusters = sum(np.unique(clustering_result)>=0)
            if n_clusters > 1:
                # evaluate the clustering
                wandb.log({"silhouette_score": silhouette_score(embeddings_red, clustering_result)})
                wandb.log({"calinski_harabasz_score": calinski_harabasz_score(embeddings_red, clustering_result)})
                wandb.log({"davies_bouldin_score": davies_bouldin_score(embeddings_red, clustering_result)})
                # log cluster information
                wandb.log({"n_clusters": n_clusters})
                cluster_sizes = pd.Series(clustering_result).value_counts().reset_index()
                wandb.log({"cluster_sizes": wandb.Table(dataframe=cluster_sizes)})

                # save images per cluster
                for cl_label, cl_examples in pd.DataFrame({'cluster': clustering_result, 'original_pos': original_pos}).groupby('cluster'):
                    # get the original image positions
                    if cl_label >= 0:
                        mask = np.full(n_images, False)
                        mask[cl_examples['original_pos']] = True
                        wandb.log({"cluster_images": wandb.Image(images[mask], caption=f"{job_name} | Label {cl_label} | (N = {cl_examples.shape[0]})")})

                # get prototypes of each cluster
                proto_idx = None
                prototypes = None
                if cl_name == 'hdbscan':
                    prototypes = cl_method.medoids_
                elif cl_name == 'dbscan':
                    proto_idx = cl_method.core_sample_indices_
                elif cl_name == 'kmeans':
                    means = cl_method.get_centers()
                    proto_idx = [find_closest_point(mean_point, embeddings_red) for mean_point in means]
                else:
                    # calculate the medoid per each cluster whose label is >= 0
                    prototypes = [calculate_medoid(embeddings_red[clustering_result == cl_label]) for cl_label in np.unique(clustering_result) if cl_label >= 0]
                    
                if (prototypes is not None) & (proto_idx is None):
                    # find centroids in the original data and get the indice
                    proto_idx = [np.where(np.all(embeddings_red == el, axis=1))[0][0] for el in prototypes]

                # save images
                if proto_idx is not None:
                    # referencing to the original images
                    mask = np.full(n_images, False)
                    mask[original_pos[proto_idx]] = True
                    wandb.log({"prototypes": wandb.Image(images[mask], caption=f"{job_name}")})

                # TODO: create visualizations
                # TODO: colors per cluster
                # TODO: colors per prototype    
                
            # close wandb - after each clustering
            wandb.finish()


if __name__ == "__main__":
    create_cluster_image()
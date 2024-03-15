"""
python -m src.clustering.test --config experiments/patterns/mnist_7v1.yml --run_id 0hvkl8kz --dim_red umap_80 --clustering gmm_d3
python -m src.clustering.test --config experiments/patterns/mnist_5v3.yml --run_id a3f602un --dim_red umap_10 --clustering gmm_s4
python -m src.clustering.test --config experiments/patterns/mnist_8v0.yml --run_id qazkm46b --dim_red umap_80 --clustering gmm_s3
python -m src.clustering.test --config experiments/patterns/mnist_9v4.yml --run_id lxshxwgn --dim_red umap_80 --clustering gmm_s3
python -m src.clustering.test --config experiments/patterns/fashion_3v0.yml --run_id dux9zf1i --dim_red umap_80 --clustering gmm_s3
"""
import argparse
import os
import numpy as np
import wandb
from dotenv import load_dotenv
from sklearn.cluster import HDBSCAN, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from src.utils.config import read_config
from src.clustering.aux import calculate_medoid, calculate_test_embeddings, find_closest_point, create_wandb_report_metrics, create_wandb_report_images, create_wandb_report_2dviz
import torch 
#from pyclustering.cluster.xmeans import xmeans


# available reductions and clustering options to test in the pipeline
REDUCTION_DICT = {
    'None': None,
    'pca_90': PCA(n_components=0.9),
    'pca_70': PCA(n_components=0.7),
    'tsne_3': TSNE(n_components=3),
    'umap_80': UMAP(n_components=80, n_neighbors=8, min_dist=0.05, random_state=0),
    'umap_10': UMAP(n_components=10, n_neighbors=8, min_dist=0.05, random_state=0),
}
# step 2 - clustering
CLUSTERING_DICT = {
    'dbscan': DBSCAN(min_samples=5, eps=0.2),
    'hdbscan': HDBSCAN(min_samples=5, store_centers="both"),
    #'kmeans': None,
    'ward': AgglomerativeClustering(distance_threshold=25, n_clusters=None),
    'gmm_s3': GaussianMixture(n_components=3, covariance_type='spherical', random_state=0),
    'gmm_s4': GaussianMixture(n_components=4, covariance_type='spherical', random_state=0),
    'gmm_d3': GaussianMixture(n_components=3, covariance_type='diag', random_state=0),
}

def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        help="Config file", required=True)
    parser.add_argument("--run_id", dest="run_id",
                        help="Experiment ID (seen in wandb) from GAN", required=True)
    parser.add_argument("--dim_red", dest="dim_red",
                        help="Dimensionality reduction method", required=False)
    parser.add_argument("--clustering", dest="clustering",
                        help="Clustering method", required=False)
    return parser.parse_args()

def create_cluster_image(config, run_id, dim_red=None, clustering=None):
    """_summary_
    """
    # initialize variables
    config_run = {}
    device = config["device"]
    DIR = f"{os.environ['FILESDIR']}/data/clustering/{run_id}"
    # the embeddings and the images are saved in the same order
    C_emb = torch.load(f"{DIR}/classifier_embeddings.pt")
    images = torch.load(f"{DIR}/images_acd_1.pt").to(device)
   
    my_clusterings = CLUSTERING_DICT
    my_reductions = REDUCTION_DICT
    if (dim_red in REDUCTION_DICT) & (clustering in CLUSTERING_DICT):
        my_reductions = {dim_red: REDUCTION_DICT[dim_red]}
        my_clusterings = {clustering: CLUSTERING_DICT[clustering]}
    else:
        print("all vs all approach...")

    # get embeddings
    with torch.no_grad():
        embeddings = C_emb(images)

    # get test set images
    embeddings_tst, preds = calculate_test_embeddings(config["dataset"]["name"], config["data-dir"], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], config['train']['step-2']['batch-size'], device, C_emb)

    # one wandb run for each clustering
    for cl_name, cl_method in my_clusterings.items():
        for red_name, red_method in my_reductions.items():
            # start wandb
            config_run['clustering_method'] = cl_name
            config_run['reduce_method'] = red_name

            job_name = f"{cl_name}_{red_name}" if red_name != "None" else cl_name
            wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-4-clustering_{job_name}',
                name=f"{run_id}_v3",
                config=config_run)
            
            # apply reduction method
            embeddings_red = red_method.fit_transform(embeddings.cpu().detach().numpy()) if red_name != "None" else embeddings
            # scikit-learn methods
            clustering_result = cl_method.fit_predict(embeddings_red)

            # if string includes 'kmeans' then apply xmeans
            # I think that in this case we cannot guarante the order of the clusters
            #if cl_name == 'kmeans' :
            #    # define here the instance
            #    cl_method = xmeans(embeddings_red, k_max=15)
            #    cl_method.process()
            #    clustering_xmeans = cl_method.get_clusters()
            #    subcluster_labels = {subcluster_index: i for i, x_cluster in enumerate(clustering_xmeans) for subcluster_index in x_cluster}
            #    clustering_result = [x_label for _, x_label in sorted(subcluster_labels.items())] 
 
            # verify if it worked
            n_clusters = sum(np.unique(clustering_result)>=0)
            if n_clusters <= 1:
                print("not possible to cluster")
            else:
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
                    proto_idx_torch = torch.tensor(proto_idx).to(device)
                
                    # create wandb report
                    create_wandb_report_metrics(wandb, embeddings_red, clustering_result)
                    create_wandb_report_images(wandb, job_name, images, clustering_result, proto_idx_torch)
                    create_wandb_report_2dviz(wandb, job_name, embeddings, embeddings_tst, proto_idx_torch, preds, clustering_result)
                
            # close wandb - after each clustering
            wandb.finish()

if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    config = read_config(args.config)
    create_cluster_image(config, args.run_id, args.dim_red, args.clustering)

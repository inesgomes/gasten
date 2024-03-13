import argparse
import os
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.cluster import HDBSCAN, DBSCAN, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from src.utils.config import read_config
from src.clustering.aux import calculate_medoid, find_closest_point
from src.datasets import load_dataset
import torch 
import matplotlib.pyplot as plt
#from pyclustering.cluster.xmeans import xmeans


# available reductions and clustering options to test in the pipeline
REDUCTION_DICT = {
    'None': None,
    'pca_90': PCA(n_components=0.9),
    'pca_70': PCA(n_components=0.7),
    'tsne_3': TSNE(n_components=3),
    'umap_15': UMAP(n_components=15),
}
# step 2 - clustering
CLUSTERING_DICT = {
    'dbscan': DBSCAN(min_samples=5, eps=0.2),
    'hdbscan': HDBSCAN(min_samples=5, store_centers="both"),
    #'kmeans': None,
    'ward': AgglomerativeClustering(distance_threshold=25, n_clusters=None),
    'gaussian_mixture': BayesianGaussianMixture(n_components=3, covariance_type='full', max_iter=1000, random_state=0),
}

def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        help="Config file", required=True)
    parser.add_argument("--dataset_id", dest="dataset_id",
                        help="Experiment ID (seen in wandb)", required=True)
    parser.add_argument("--dim_red", dest="dim_red",
                        help="Dimensionality reduction method", required=False)
    parser.add_argument("--clustering", dest="clustering",
                        help="Clustering method", required=False)
    return parser.parse_args()


def create_cluster_image(config, dataset_id, dim_red=None, clustering=None):
    """_summary_
    """
    # initialize variables
    config_run = {}
    DIR = f"{os.environ['FILESDIR']}/data/clustering/{dataset_id}"
    # the embeddings and the images are saved in the same order
    C_emb = torch.load(f"{DIR}/classifier_embeddings.pt")
    images = torch.load(f"{DIR}/images_acd_1.pt")
    device = config["device"]

    my_clusterings = CLUSTERING_DICT
    my_reductions = REDUCTION_DICT
    if (dim_red in REDUCTION_DICT) & (clustering in CLUSTERING_DICT):
        my_reductions = {dim_red: REDUCTION_DICT[dim_red]}
        my_clusterings = {clustering: CLUSTERING_DICT[clustering]}
    else:
        print("all vs all approach...")

    # get embeddings
    with torch.no_grad():
        embeddings = C_emb(images.to(device))

    # get test set images
    test_set = load_dataset(config["dataset"]["name"], config["data-dir"], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], train=False)[0]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['train']['step-2']['batch-size'], shuffle=False)
    embeddings_tst_array = []
    with torch.no_grad():
        for data_tst in test_loader:
            X, _ = data_tst
            embeddings_tst_array.append(C_emb(X.to(device)))
    # concatenate the array
    embeddings_tst = torch.cat(embeddings_tst_array, dim=0)

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
                name=f"{dataset_id}_v3",
                config=config_run)
            
            # apply reduction method
            embeddings_red = red_method.fit_transform(embeddings) if red_name != "None" else embeddings
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
                for cl_label, cl_examples in pd.DataFrame({'cluster': clustering_result, 'image': images}).groupby('cluster'):
                    # get the original image positions
                    if cl_label >= 0:
                        wandb.log({"cluster_images": wandb.Image(cl_examples, caption=f"{job_name} | Label {cl_label} | (N = {cl_examples.shape[0]})")})

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
                    wandb.log({"prototypes": wandb.Image(images[proto_idx], caption=f"{job_name}")})

                # TSNE visualization
                # merge tst and ambiguous examples
                emb_tst_protos = torch.cat([embeddings_tst, embeddings[proto_idx]], dim=0)
                final_red = TSNE(n_components=2).fit_transform(emb_tst_protos.cpu().detach().numpy())

                red_1 = final_red[:embeddings_tst.shape[0]]
                red2 = final_red[embeddings_tst.shape[0]:]
                plt.scatter(x=red_1[:, 0], y=red_1[:, 1], marker='o', label='test set')
                plt.scatter(x=red_2[:, 0], y=red_2[:, 1], marker='x', label='prototypes')
                plt.legend()
                wandb.log({f"Embeddings (test set and prototypes)": wandb.Image(plt)})
                plt.close()
                # get the test set and color it with red/green according to positive/negative class
                # get the ambiguous images and color them according to the cluster (with and without the test set)
                # test set + prototypes
                # ambiguous + prototypes
                # TODO: colors per cluster
                # TODO: colors per prototype    
                
            # close wandb - after each clustering
            wandb.finish()


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    config = read_config(args.config)

    create_cluster_image(config, args.dataset_id, args.dim_red, args.clustering)

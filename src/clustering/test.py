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

def viz_2d_test_prototypes(viz_embeddings, n, preds, name):
    """_summary_
    get the test set and color it with red/green according to positive/negative class
    mark the prototypes with a black x
    Args:
        viz_embeddings (_type_): _description_
        n (_type_): _description_
        preds (_type_): _description_

    Returns:
        _type_: _description_
    """
    neg_emb = viz_embeddings[:n][preds==0]
    pos_emb = viz_embeddings[:n][preds==1]
    proto_emb = viz_embeddings[n:]
    
    plt.figure(figsize=(9,8))
    plt.scatter(x=neg_emb[:, 0], y=neg_emb[:, 1], marker='*', label='negative (test set)', c='firebrick', alpha=0.1)
    plt.scatter(x=pos_emb[:, 0], y=pos_emb[:, 1], marker='*', label='positive (test set)', c='green', alpha=0.1)
    plt.scatter(x=proto_emb[:, 0], y=proto_emb[:, 1], marker='X', label='prototypes', c='black')
    plt.title(f"{name} 2D")
    plt.legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small')
    return plt

def viz_2d_ambiguous_prototypes(viz_embeddings, n, clustering_result, name):
    """_summary_
    only ambiguous images colored per cluster
    prototypes with a black x
    Args:
        viz_embeddings (_type_): _description_
        n (_type_): _description_
        clustering_result (_type_): _description_

    Returns:
        _type_: _description_
    """
    amb_1 = viz_embeddings[:n]
    amb_2 = viz_embeddings[n:]
    plt.figure(figsize=(9,8))
    plt.scatter(x=amb_1[:, 0], y=amb_1[:, 1], c=clustering_result, cmap='Set1', alpha=0.8, label='ambiguous images clusters', s=3)
    plt.scatter(x=amb_2[:, 0], y=amb_2[:, 1], marker='X', label='prototypes', c='black')
    plt.legend(ncols=2, bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    plt.title(f"{name} 2D")
    plt.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small')
    return plt

def viz_2d_all(viz_embeddings, n_tst, n_protos, preds, clustering_result, name):
    """_summary_

    Args:
        viz_embeddings (_type_): _description_
        n_tst (_type_): _description_
        n_protos (_type_): _description_
        preds (_type_): _description_
        clustering_result (_type_): _description_

    Returns:
        _type_: _description_
    """
    emb_all_tst_pos = viz_embeddings[:n_tst][preds==1]
    emb_all_tst_neg = viz_embeddings[:n_tst][preds==0]
    emb_all_amb = viz_embeddings[n_tst:-n_protos]
    emb_all_proto = viz_embeddings[-n_protos:]
    
    plt.figure(figsize=(9,8))
    plt.scatter(x=emb_all_tst_neg[:, 0], y=emb_all_tst_neg[:, 1], alpha=0.1, label='negative (test set)', marker="*", color="green")
    plt.scatter(x=emb_all_tst_pos[:, 0], y=emb_all_tst_pos[:, 1], alpha=0.1, label='positive (test set)', marker="*", color="firebrick")
    plt.scatter(x=emb_all_amb[:, 0], y=emb_all_amb[:, 1], c=clustering_result, cmap='Set1', alpha=0.8, label='ambiguous images clusters', s=3)
    plt.scatter(x=emb_all_proto[:, 0], y=emb_all_proto[:, 1], marker='X', label='prototypes', c='black')
    plt.title(f"{name} 2D")
    plt.legend(ncols=4, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small')
    
    return plt

def create_cluster_image(config, dataset_id, dim_red=None, clustering=None):
    """_summary_
    """
    # initialize variables
    config_run = {}
    device = config["device"]
    DIR = f"{os.environ['FILESDIR']}/data/clustering/{dataset_id}"
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
    test_set = load_dataset(config["dataset"]["name"], config["data-dir"], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], train=False)[0]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['train']['step-2']['batch-size'], shuffle=False)
    embeddings_tst_array = []
    preds = []
    with torch.no_grad():
        for data_tst in test_loader:
            X, y = data_tst
            embeddings_tst_array.append(C_emb(X.to(device)))
            preds.append(y)

    # concatenate the array
    embeddings_tst = torch.cat(embeddings_tst_array, dim=0)
    preds = torch.cat(preds, dim=0).cpu().detach().numpy()

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
                for cl_label, example_no in pd.DataFrame({'cluster': clustering_result, 'image': np.arange(clustering_result.shape[0])}).groupby('cluster'):
                    # get the original image positions
                    if cl_label >= 0:
                        selected_images = torch.index_select(images, 0, torch.tensor(list(example_no["image"])).to(device))
                        wandb.log({"cluster_images": wandb.Image(selected_images, caption=f"{job_name} | Label {cl_label} | (N = {len(example_no)})")})

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
                    selected_images = torch.index_select(images, 0, torch.tensor(proto_idx).to(device))
                    wandb.log({"prototypes": wandb.Image(selected_images, caption=f"{job_name}")})

                # TSNE visualizations - merge everything and plot
                alg_tsne = TSNE(n_components=2)
                tsne = "TSNE"
                # UMAP visualizations - train on test
                alg_umap = UMAP(n_components=2).fit(embeddings_tst.cpu().detach().numpy())
                umap = "UMAP"

                # test set + prototypes
                emb_tst_protos = torch.cat([embeddings_tst, embeddings[proto_idx]], dim=0)
                # tsne 
                final_red = alg_tsne.fit_transform(emb_tst_protos.cpu().detach().numpy())
                wandb.log({
                    f"{tsne} 2D embeddings - test set + clustering prototypes": 
                    wandb.Image(viz_2d_test_prototypes(final_red, embeddings_tst.shape[0], preds, tsne))
                })
                # umap
                final_red = alg_umap.fit_transform(emb_tst_protos.cpu().detach().numpy())
                wandb.log({
                    f"{umap} 2D embeddings - test set + clustering prototypes": 
                    wandb.Image(viz_2d_test_prototypes(final_red, embeddings_tst.shape[0], preds, umap))
                })
                    
                # ambiguous images + prototypes
                emb_amb_protos = torch.cat([embeddings, embeddings[proto_idx]], dim=0)
                # tsne
                ambiguous_cl = alg_tsne.fit_transform(emb_amb_protos.cpu().detach().numpy())
                wandb.log({
                    f"{tsne} 2D embeddings - synthetic images clustering + prototypes": 
                    wandb.Image(viz_2d_ambiguous_prototypes(ambiguous_cl, embeddings.shape[0], clustering_result, tsne))
                })
                # umap
                ambiguous_cl = alg_umap.transform(emb_amb_protos.cpu().detach().numpy())
                wandb.log({
                    f"{umap} 2D embeddings - synthetic images clustering + prototypes": 
                    wandb.Image(viz_2d_ambiguous_prototypes(ambiguous_cl, embeddings.shape[0], clustering_result, umap))
                })

                # test set + ambiguous + prototypes
                emb_all = torch.cat([embeddings_tst, embeddings, embeddings[proto_idx]], dim=0)
                # tsne
                emb_all_red = alg_tsne.fit_transform(emb_all.cpu().detach().numpy())
                wandb.log({
                    f"{tsne} 2D embeddings - test set + clustering and prototypes": 
                    wandb.Image(viz_2d_all(emb_all_red, embeddings_tst.shape[0], len(proto_idx), preds, clustering_result, tsne))
                })
                # umap
                emb_all_red = alg_umap.transform(emb_all.cpu().detach().numpy())
                wandb.log({
                    f"{umap} 2D embeddings - test set + clustering and prototypes": 
                    wandb.Image(viz_2d_all(emb_all_red, embeddings_tst.shape[0], len(proto_idx), preds, clustering_result, umap))
                })
            
                
            # close wandb - after each clustering
            wandb.finish()


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    config = read_config(args.config)

    create_cluster_image(config, args.dataset_id, args.dim_red, args.clustering)

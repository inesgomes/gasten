import argparse
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import torch
from umap import UMAP
from src.datasets import load_dataset
import wandb


def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        required=True, help="Config file from experiments/clustering folder")
    return parser.parse_args()

def get_clustering_path(clustering_path, run_id, classifier):
    path = f"{clustering_path}/{run_id}/{classifier}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_gan_path(config, classifier_name):
    """_summary_

    Args:
        config (_type_): _description_
        classifier (_str_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    run_id = config["gasten"]["run-id"]
    project = config["project"]
    name = config["name"]
    # find directory whose name ends with a given id
    for dir in os.listdir(f"{os.environ['FILESDIR']}/out/{project}/{config['name']}"):
        if dir.endswith(run_id):
            return f"{os.environ['FILESDIR']}/out/{project}/{name}/{dir}/{classifier_name.split(".")[0]}_{config['gasten']['weight']}_{config['gasten']['epoch']['step-1']}/{config['gasten']['epoch']['step-2']}"

    raise Exception(f"Could not find directory with id {run_id}")
    

def calculate_test_embeddings(dataset_name, data_dir, pos, neg, batch_size, device, C_emb):
    test_set = load_dataset(dataset_name, data_dir, pos, neg, train=False)[0]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)
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

    return embeddings_tst, preds

def euclidean_distance(point1, point2):
    """_summary_

    Args:
        point1 (_type_): _description_
        point2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.sqrt(np.sum((point1 - point2)**2))


def find_closest_point(target_point, dataset):
    """_summary_

    Args:
        target_point (_type_): _description_
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        cluster_points (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Calculate pairwise distances
    distances = cdist(cluster_points, cluster_points, metric='euclidean')
    # Find the index of the point with the smallest sum of distances
    medoid_index = np.argmin(np.sum(distances, axis=0))
    # Retrieve the medoid point
    return cluster_points[medoid_index]


def gmm_bic_score(estimator, X):
    """_summary_
    Callable to pass to GridSearchCV that will use the BIC score.
    Args:
        estimator (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Make it negative since GridSearchCV expects a score to maximize
    print(estimator)
    return -estimator[1].bic(X)

def sil_score(estimator, X):
    """_summary_

    Args:
        estimator (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_red = estimator[0].fit_transform(X)
    labels = estimator[1].fit_predict(x_red)
    return silhouette_score(x_red, labels)

def db_score(estimator, X):
    """_summary_

    Args:
        estimator (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_red = estimator[0].fit_transform(X)
    labels = estimator[1].fit_predict(x_red)
    return -davies_bouldin_score(x_red, labels)

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
    plt.title(name)
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
    plt.title(name)
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
    plt.scatter(x=emb_all_tst_neg[:, 0], y=emb_all_tst_neg[:, 1], alpha=0.1, label='negative (test set)', marker="*", color="firebrick")
    plt.scatter(x=emb_all_tst_pos[:, 0], y=emb_all_tst_pos[:, 1], alpha=0.1, label='positive (test set)', marker="*", color="green")
    plt.scatter(x=emb_all_amb[:, 0], y=emb_all_amb[:, 1], c=clustering_result, cmap='Set1', alpha=0.8, label='ambiguous images clusters', s=3)
    plt.scatter(x=emb_all_proto[:, 0], y=emb_all_proto[:, 1], marker='X', label='prototypes', c='black')
    plt.title(name)
    plt.legend(ncols=4, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small')
    return plt

def create_wandb_report_metrics(embeddings_red, clustering_result):
    # evaluate the clustering
    wandb.log({"silhouette_score": silhouette_score(embeddings_red, clustering_result)})
    wandb.log({"calinski_harabasz_score": calinski_harabasz_score(embeddings_red, clustering_result)})
    wandb.log({"davies_bouldin_score": davies_bouldin_score(embeddings_red, clustering_result)})
    # log cluster information
    wandb.log({"n_clusters": sum(np.unique(clustering_result)>=0)})
    cluster_sizes = pd.Series(clustering_result).value_counts().reset_index()
    wandb.log({"cluster_sizes": wandb.Table(dataframe=cluster_sizes)})

def create_wandb_report_images(job_name, images, clustering_result, proto_idx, device='cuda:0'):
    # save images per cluster
    for cl_label, example_no in pd.DataFrame({'cluster': clustering_result, 'image': np.arange(clustering_result.shape[0])}).groupby('cluster'):
        # get the original image positions
        if cl_label >= 0:
            selected_images = torch.index_select(images, 0, torch.tensor(list(example_no["image"])).to(device))
            wandb.log({"cluster_images": wandb.Image(selected_images, caption=f"{job_name} | Label {cl_label} | (N = {len(example_no)})")})

    # save prototypes        
    selected_images = torch.index_select(images, 0, proto_idx)
    wandb.log({"prototypes": wandb.Image(selected_images, caption=job_name)})

def create_wandb_report_2dviz(job_name, embeddings, embeddings_tst, proto_idx, preds, clustering_result):
    # TSNE visualizations - merge everything and plot
    alg_tsne = TSNE(n_components=2)
    tsne = "TSNE "
    # UMAP visualizations - train on test
    alg_umap = UMAP(n_components=2).fit(embeddings_tst.cpu().detach().numpy())
    umap = "UMAP "

    prototypes = torch.index_select(embeddings, 0, proto_idx)

    # test set + prototypes
    emb_tst_protos = torch.cat([embeddings_tst, prototypes], dim=0)
    title = "2D embeddings - test set + clustering prototypes"
    # tsne 
    final_red = alg_tsne.fit_transform(emb_tst_protos.cpu().detach().numpy())
    wandb.log({
        tsne+title: 
        wandb.Image(viz_2d_test_prototypes(final_red, embeddings_tst.shape[0], preds, job_name))
    })

    # umap
    final_red = alg_umap.fit_transform(emb_tst_protos.cpu().detach().numpy())
    wandb.log({
        umap+title: 
        wandb.Image(viz_2d_test_prototypes(final_red, embeddings_tst.shape[0], preds, job_name))
    })
                    
    # ambiguous images + prototypes
    emb_amb_protos = torch.cat([embeddings, embeddings[proto_idx]], dim=0)
    title = "2D embeddings - synthetic images clustering + prototypes"
    # tsne
    ambiguous_cl = alg_tsne.fit_transform(emb_amb_protos.cpu().detach().numpy())
    wandb.log({
        tsne + title: 
        wandb.Image(viz_2d_ambiguous_prototypes(ambiguous_cl, embeddings.shape[0], clustering_result, job_name))
    })
    # umap
    ambiguous_cl = alg_umap.transform(emb_amb_protos.cpu().detach().numpy())
    wandb.log({
        umap+title: 
        wandb.Image(viz_2d_ambiguous_prototypes(ambiguous_cl, embeddings.shape[0], clustering_result, job_name))
    })

    # test set + ambiguous + prototypes
    emb_all = torch.cat([embeddings_tst, embeddings, embeddings[proto_idx]], dim=0)
    title = "2D embeddings - test set + clustering and prototypes"
    # tsne
    emb_all_red = alg_tsne.fit_transform(emb_all.cpu().detach().numpy())
    wandb.log({
        tsne+title: 
        wandb.Image(viz_2d_all(emb_all_red, embeddings_tst.shape[0], len(proto_idx), preds, clustering_result, job_name))
    })
    # umap
    emb_all_red = alg_umap.transform(emb_all.cpu().detach().numpy())
    wandb.log({
        umap+title: 
        wandb.Image(viz_2d_all(emb_all_red, embeddings_tst.shape[0], len(proto_idx), preds, clustering_result, job_name))
    })


def visualize_embeddings(config, C_emb, pred_syn, embeddings_f):
    """
    """
    device = config["device"]

    # get test set 
    test_set = load_dataset(config["dataset"]["name"], config["dir"]["data"], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], train=False)[0]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

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

    # prepare viz
    print("Start visualizing embeddings...")
    alpha = 0.7
    cmap = 'RdYlGn'

    viz_algs = {
        'PCA': PCA(n_components=2),
        'UMAP': UMAP(n_components=2),
        'TSNE': TSNE(n_components=2),
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

        if name == 'TSNE':
            red_embs_test = alg.fit_transform(embeddings_tst_cpu)
        else:
            alg_tst = alg.fit(embeddings_tst_cpu)
            red_embs_test = alg_tst.transform(embeddings_tst_cpu)
        plt.scatter(x=red_embs_test[:, 0], y=red_embs_test[:, 1], c=pred_tst, cmap=cmap, marker='x', vmin=0, vmax=1)
        wandb.log({f"{name} Embeddings (test set)": wandb.Image(plt)})
        plt.close()

        if name == 'TSNE':
            red_embs_total = alg.fit_transform(embeddings_total)
        else:
            red_embs_total = alg_tst.transform(embeddings_total)
        real_embs = red_embs_total[:size_real]
        syn_embs = red_embs_total[size_real:]

        plt.scatter(real_embs[:, 0], real_embs[:, 1], c=pred_tst, label='Real Data', cmap=cmap, alpha=alpha, marker='x', vmin=0, vmax=1)
        plt.scatter(syn_embs[:, 0], syn_embs[:, 1], c=pred_syn, label='Synthetic Data', cmap=cmap, alpha=0.5, marker='o', vmin=0, vmax=1)
        plt.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small')
        wandb.log({f"{name} Embeddings (test set + gen)": wandb.Image(plt)})
        plt.close()

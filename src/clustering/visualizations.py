import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.datasets import load_dataset
from umap import UMAP
import torch
import wandb

# Create a colormap from the list
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom", ["#ffbc3f", "#0c28e9", "#ff714b", "#754200", "#ff0079", "#962400", "#358fa1", "#f9f871", "#d900b4", "#8676a9", "#7a7485", "#342c49", ])

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
    plt.scatter(x=proto_emb[:, 0], y=proto_emb[:, 1], marker='x', label='prototypes', c='red')
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
    plt.scatter(x=amb_1[:, 0], y=amb_1[:, 1], c=clustering_result, cmap=CUSTOM_CMAP, alpha=0.8, label='ambiguous clusters')
    plt.scatter(x=amb_2[:, 0], y=amb_2[:, 1], marker='x', label='prototypes', c='red')
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
    plt.scatter(x=emb_all_amb[:, 0], y=emb_all_amb[:, 1], c=clustering_result, cmap=CUSTOM_CMAP, label='ambiguous clusters', alpha=0.8, s=10)
    plt.scatter(x=emb_all_proto[:, 0], y=emb_all_proto[:, 1], marker='x', label='prototypes', c='red')
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

def create_wandb_report_prototypes(job_name, images, proto_idx):
    # save prototypes        
    selected_images = torch.index_select(images, 0, proto_idx)
    wandb.log({"prototypes": wandb.Image(selected_images, caption=job_name)})

def create_wandb_report_images(job_name, images, clustering_result, device='cuda:0'):
    # save images per cluster
    for cl_label, example_no in pd.DataFrame({'cluster': clustering_result, 'image': np.arange(clustering_result.shape[0])}).groupby('cluster'):
        # get the original image positions
        if cl_label >= 0:
            selected_images = torch.index_select(images, 0, torch.tensor(list(example_no["image"])).to(device))
            wandb.log({"cluster_images": wandb.Image(selected_images, caption=f"{job_name} | Label {cl_label} | (N = {len(example_no)})")})

def prepare_2dvisualization(ambiguous, embeddings, colors, prototypes, alg, name, job_name):

    emb_protos = torch.cat([embeddings, prototypes], dim=0).cpu().detach().numpy()
    final_red = alg.fit_transform(emb_protos) if name=="tsne" else alg.transform(emb_protos)

    if ambiguous:
        title = "2D embeddings - clustering + prototypes"
        plt = viz_2d_ambiguous_prototypes(final_red, embeddings.shape[0], colors, job_name)
    else:
        title = "2D embeddings - test set + prototypes"
        plt = viz_2d_test_prototypes(final_red, embeddings.shape[0], colors, job_name)
        
    wandb.log({
        name + title:
        wandb.Image(plt)
    })
    plt.close()

def prepare_2dvisualization_all(embeddings_tst, embeddings, prototypes, y_test, clustering_result, alg, name, job_name):
    # test set + ambiguous + prototypes
    emb_all = torch.cat([embeddings_tst, embeddings, prototypes], dim=0).cpu().detach().numpy()
    title = "2D embeddings - test set + clustering + prototypes"

    emb_all_red = alg.fit_transform(emb_all) if name == "tsne" else alg.transform(emb_all)
    plt = viz_2d_all(emb_all_red, embeddings_tst.shape[0], prototypes.shape[0], y_test, clustering_result, job_name)
    wandb.log({
        name+title:
        wandb.Image(plt)
    })
    plt.close()


def create_wandb_report_2dviz(job_name, embeddings, clustering_result, proto_idx, embeddings_tst, y_test):

    algs = {
        'tsne': TSNE(n_components=2),
        'umap': UMAP(n_components=2).fit(embeddings_tst.cpu().detach().numpy()),
    }
    prototypes = torch.index_select(embeddings, 0, proto_idx)

    for name, alg in algs.items():
        prepare_2dvisualization(False, embeddings_tst, y_test, prototypes, alg, name, job_name)
        if embeddings.shape[0]>30:
            prepare_2dvisualization(True, embeddings, clustering_result, prototypes, alg, name, job_name)
        prepare_2dvisualization_all(embeddings_tst, embeddings, prototypes, y_test, clustering_result, alg, name, job_name)
        


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

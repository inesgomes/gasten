import os
from dotenv import load_dotenv
import numpy as np
import torch
from src.utils.config import read_config_clustering
from src.clustering.aux import parse_args, get_clustering_path, calculate_test_embeddings
from src.clustering.visualizations import create_wandb_report_images, create_wandb_report_2dviz, create_wandb_report_prototypes, prepare_2dvisualization
from src.clustering.optimize import load_gasten_images
from src.clustering.generate_embeddings import load_gasten
from src.datasets import load_dataset
import wandb
from sklearn.metrics.pairwise import cosine_similarity
from captum.attr import Saliency, GradientShap
from captum.attr import visualization as viz
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from umap import UMAP


def transform_original_image(image):
    return np.transpose((image.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

def saliency_maps(clf, images, device):
    """
    1st criteria  - interpretability
    This function generates saliency maps for the prototypes (with captum)
    """
    reference_input = torch.full(images[0].shape, -1).unsqueeze(0).to(device)

    for ind, image in enumerate(images):
        input = image.unsqueeze(0)
        input.requires_grad = True
        original_image = transform_original_image(image)
        # compute gradient shap
        with torch.no_grad():
            feature_imp_img = GradientShap(clf).attribute(input, baselines=reference_input)
        attr = feature_imp_img.squeeze(0).cpu().detach().numpy().reshape(28, 28, 1)
        # visualization
        my_viz, _ = viz.visualize_image_attr(attr, original_image, method="blended_heat_map",
                             sign="all", show_colorbar=True, use_pyplot=False)
        wandb.log({"gradient_shap": wandb.Image(my_viz, caption=f"prototype {ind}")})

        # saliency map
        grads = Saliency(clf).attribute(input)
        grads = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # visualization
        my_viz, _ = viz.visualize_image_attr(grads, original_image,  method="blended_heat_map",
                                             sign="absolute_value", show_colorbar=True, use_pyplot=False)
        wandb.log({"saliency_maps": wandb.Image(my_viz, caption=f"prototype {ind}")})

def diversity_apd(embeddings, proto_idx):
    """
    2nd criteria: diversity
    average pairwise distance: similarity among all images within a single set
    """
    prototypes = torch.index_select(embeddings, 0, proto_idx).cpu().detach().numpy()

    similarity_matrix = cosine_similarity(np.array(prototypes))
    # Since the matrix includes similarity of each image with itself (1.0), we'll zero these out for a fair average
    np.fill_diagonal(similarity_matrix, 0)
    # Calculate the average similarity, excluding self-similarities
    return np.sum(similarity_matrix) / (similarity_matrix.size - len(similarity_matrix))

def coverage():
    """
    TODO
    3rd criteria: coverage of the DB
    """
    pass

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

def load_estimator(config, classifier_name, dim_reduction, clustering, embeddings):
    """
    """
    estimator_name = f"{dim_reduction}_{clustering}"
    # load estimator and calculate the embeddings and clustering_result
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    estimator = torch.load(f"{path}/{estimator_name}.pt")
    # predict
    embeddings_cpu = embeddings.detach().cpu().numpy()
    embeddings_red = estimator[0].fit_transform(embeddings_cpu)
    clustering_results = estimator[1].fit_predict(embeddings_red)
    return embeddings_red, clustering_results

def baseline_prototypes(config, classifier_name, C, C_emb, n_samples=10, iter=0):
    """
    This function calculates the prototypes of the baseline
    """
    device = config["device"]

    config_run = {
        'step': 'baseline_prototypes',
        'classifier_name': classifier_name,
        'n_samples': n_samples
    }
    # prepare wandb job
    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type='step-5-baseline',
                name=f"{config['gasten']['run-id']}-{classifier_name}_{config['tag']}-{iter}",
                tags=[config["tag"]],
                config=config_run
            )

    print("> Extracting test set ...")
    test_set = load_dataset(config['dataset']['name'], config['dir']['data'], config['dataset']['binary']['pos'], config['dataset']['binary']['neg'], train=False)[0]
    test_loader = torch.utils.data.DataLoader(test_set, config['batch-size'], shuffle=False)
    preds = []
    y_test = []
    embeddings = []
    images = []
    with torch.no_grad():
        for data_tst in test_loader:
            X, y = data_tst
            images.append(X.to(device))
            preds.append(C(X.to(device)))
            y_test.append(y)
            embeddings.append(C_emb(X.to(device)))

    # concatenate the array
    preds = torch.cat(preds, dim=0).cpu().detach().numpy()
    y_test = torch.cat(y_test, dim=0).cpu().detach().numpy()
    embeddings = torch.cat(embeddings, dim=0)
    images = torch.cat(images, dim=0)
    # filter by ACD
    print("> Selecting images ...")
    mask = (preds >= (0.5 - config["clustering"]["acd"])) & (preds <= (0.5 + config["clustering"]["acd"]))
    embeddings_mask = embeddings[mask]
    images_mask = images[mask]
    proto_idx_torch = torch.tensor(np.random.choice(range(0, mask.sum()), size=n_samples, replace=False)).to(device)

    # evaluate - same as prototypes
    print("> Evaluating ...")
    wandb.log({"avg_pairwise_distance": diversity_apd(embeddings_mask, proto_idx_torch)})
    selected_images = torch.index_select(images_mask, 0, proto_idx_torch)
    saliency_maps(C, selected_images, device)
    
    # visualizations
    print("> Creating visualizations...")
    create_wandb_report_prototypes(classifier_name, images_mask, proto_idx_torch)
    prototypes = torch.index_select(embeddings_mask, 0, proto_idx_torch)
    prepare_2dvisualization(embeddings, y_test, prototypes, TSNE(n_components=2), "tsne", "")
    prepare_2dvisualization(embeddings, y_test, prototypes, UMAP(n_components=2), "umap", "")
    
    wandb.finish()

def calculate_prototypes(config, typ, classifier_name, estimator_name, C, C_emb, syn_images, embeddings_ori, embeddings_red, clustering_result):
    """
    This function calculates the prototypes of each cluster
    """
    device = config["device"]

    config_run = {
        'step': 'clustering_prototypes',
        'classifier_name': classifier_name,
        'eestimator_name': estimator_name,
        'prototype_type': typ
    }

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-5-prototypes_{typ}_{estimator_name}',
                name=f"{config['gasten']['run-id']}-{classifier_name}_{config['tag']}",
                tags=[config["tag"]],
                config=config_run
                )
    
    # get prototypes of each cluster
    print("> Calculating prototypes ...")
    if typ == "medoid":
        prototypes = [calculate_medoid(embeddings_red[clustering_result == cl_label]) for cl_label in np.unique(clustering_result) if cl_label >= 0]
    elif typ == "random":
        prototypes = [np.random.choice(embeddings_red[clustering_result == cl_label]) for cl_label in np.unique(clustering_result) if cl_label >= 0]
    elif typ == "centroid":
        # TODO: Choose the sample closest to the cluster centroid (mean) as the prototype
        raise ValueError(f"Prototype type {typ} not yet implemented")
    elif typ == "density":
        # TODO:  Select a sample that resides in the densest part of the cluster
        raise ValueError(f"Prototype type {typ} not yet implemented")
    else:
        raise ValueError(f"Not a possible value for prototype type - {typ}")
    
    # get location
    proto_idx = [np.where(np.all(embeddings_red == el, axis=1))[0][0] for el in prototypes]
    proto_idx_torch = torch.tensor(proto_idx).to(device)

    print("> Evaluating ...")
    wandb.log({"avg_pairwise_distance": diversity_apd(embeddings_ori, proto_idx_torch)})
    selected_images = torch.index_select(syn_images, 0, proto_idx_torch)
    saliency_maps(C, selected_images, device)
    
    # visualizations
    print("> Creating visualizations...")
    embeddings_tst, y_test = calculate_test_embeddings(config["dataset"]["name"], config["dir"]['data'], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], config['batch-size'], device, C_emb)
    create_wandb_report_prototypes(estimator_name, syn_images, proto_idx_torch)
    create_wandb_report_images(estimator_name, syn_images, clustering_result)
    create_wandb_report_2dviz(estimator_name, embeddings_ori, clustering_result, proto_idx_torch, embeddings_tst, y_test)

    wandb.finish()


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    config = read_config_clustering(args.config)

    for classifier in config['gasten']['classifier']:
        _, C, classifier_name = load_gasten(config, classifier)
        C_emb, syn_images_f, syn_embeddings_f = load_gasten_images(config, classifier_name)
        
        for opt in config["clustering"]["options"]:
            embeddings_red, clustering_results = load_estimator(config, classifier_name, opt['dim-reduction'], opt['clustering'], syn_embeddings_f)

            for typ in config['prototypes']['type']:
                calculate_prototypes(config, typ, classifier_name, f"{opt['dim-reduction']}_{opt['clustering']}", C, C_emb, syn_images_f, syn_embeddings_f, embeddings_red, clustering_results)
                
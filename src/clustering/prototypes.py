import os
from dotenv import load_dotenv
import numpy as np
import torch
from src.utils.config import read_config_clustering
from src.clustering.aux import parse_args, get_clustering_path, calculate_medoid, create_wandb_report_images, create_wandb_report_2dviz, calculate_test_embeddings
from src.clustering.optimize import load_gasten_images
from src.clustering.generate_embeddings import load_gasten
import wandb
from sklearn.metrics.pairwise import cosine_similarity
from captum.attr import Saliency, GradientShap
from captum.attr import visualization as viz
import matplotlib.pyplot as plt


def get_x_y(index, max_y):
    return index // max_y, index % max_y

def transform_original_image(image):
    return np.transpose((image.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

def saliency_maps(clf, images):
    """
    TODO
    1st criteria  - interpretability
    This function generates saliency maps for the prototypes (with captum)
    """
    reference_input = torch.full(images[0].shape, -1).unsqueeze(0).to("cuda:0")

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

def calculate_prototypes(config, typ, classifier_name, estimator_name, C, images, embeddings_ori, embeddings_red, clustering_result):
    """
    This function calculates the prototypes of each cluster
    """
    device = config["device"]

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-5-prototypes_{typ}_{estimator_name}',
                name=f"{config['gasten']['run-id']}-{classifier_name}_v2")

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

    # TODO evaluate prototypes
    print("> Evaluating ...")
    wandb.log({"avg_pairwise_distance": diversity_apd(embeddings_ori, proto_idx_torch)})
    selected_images = torch.index_select(images, 0, proto_idx_torch)
    saliency_maps(C, selected_images)
    
    # visualizations
    print("> Creating visualizations...")
    embeddings_tst, preds = calculate_test_embeddings(config["dataset"]["name"], config["dir"]['data'], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], config['batch-size'], device, C_emb)
    create_wandb_report_images(estimator_name, images, clustering_result, proto_idx_torch)
    create_wandb_report_2dviz(estimator_name, embeddings_ori, embeddings_tst, proto_idx_torch, preds, clustering_result)

    wandb.finish()


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    config = read_config_clustering(args.config)

    for classifier in config['gasten']['classifier']:
        _, C, classifier_name = load_gasten(config, classifier)
        C_emb, images, embeddings_ori = load_gasten_images(config, classifier_name)
        
        for opt in config["clustering"]["options"]:
            embeddings_red, clustering_results = load_estimator(config, classifier_name, opt['dim-reduction'], opt['clustering'], embeddings_ori)

            for typ in config['prototypes']['type']:
                calculate_prototypes(config, typ, classifier_name, f"{opt['dim-reduction']}_{opt['clustering']}", C, images, embeddings_ori, embeddings_red, clustering_results)
                
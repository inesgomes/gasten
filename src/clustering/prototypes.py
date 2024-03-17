from dotenv import load_dotenv
import numpy as np
import os
import torch
from src.utils.config import read_config_clustering
from src.clustering.aux import parse_args, get_clustering_path, calculate_medoid, create_wandb_report_images, create_wandb_report_2dviz, calculate_test_embeddings
from src.clustering.optimize import load_gasten_images
import wandb


def saliency_maps(prototypes):
    """
    TODO
    1st criteria  - interpretability
    This function generates saliency maps for the prototypes
    """
    # apply captum saliency maps to the images
    pass

def diversity():
    """
    TODO
    - entropy: 
    - average pairwise distance
    - pixel-level (not great for few images): Calculate the standard deviation or variance for each pixel or pixel channel across all images. Higher values indicate greater diversity. 
    """
    pass


def coverage():
    """
    TODO
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


# TODO: allow for multiple types of clustering
def calculate_prototypes(config, typ, classifier_name, estimator_name, images, embeddings_ori, embeddings_red, clustering_result):
    """
    This function calculates the prototypes of each cluster
    """
    device = config["device"]

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-5-prototypes_{typ}_{estimator_name}',
                name=f"{config['gasten']['run-id']}-{classifier_name}")

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
        classifier_name = classifier.split("/")[-1]
        C_emb, images, embeddings_ori = load_gasten_images(config, classifier_name)
        
        for opt in config["clustering"]["options"]:
            embeddings_red, clustering_results = load_estimator(config, classifier_name, opt['dim-reduction'], opt['clustering'], embeddings_ori)

            for typ in config['prototypes']['type']:
                calculate_prototypes(config, typ, classifier_name, f"{opt['dim-reduction']}_{opt['clustering']}", images, embeddings_ori, embeddings_red, clustering_results)
                
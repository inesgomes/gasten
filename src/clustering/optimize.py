"""
python -m src.clustering.optimize --config experiments/patterns/mnist_5v3.yml --run_id a3f602un --epoch 10
python -m src.clustering.optimize --config experiments/patterns/mnist_8v0.yml --run_id qazkm46b --epoch 10
python -m src.clustering.optimize --config experiments/patterns/mnist_9v4.yml --run_id lxshxwgn --epoch 10
"""
import os
import argparse
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from src.clustering.aux import calculate_medoid, sil_score, create_wandb_report_metrics, create_wandb_report_images, create_wandb_report_2dviz, calculate_test_embeddings
from src.utils.config import read_config
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from umap import UMAP
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# Define the pipeline with UMAP and GMM
PIPELINE = Pipeline(steps=[
    ('umap', UMAP(random_state=2)),
    ('gmm', GaussianMixture(random_state=2)) # full -> N2D
])
# Define the parameter grid for UMAP and GMM
PARAM_SPACE = {
    'umap__n_neighbors': Integer(5, 25), #N2D: 20
    'umap__min_dist': Real(0.01, 0.5), #N2D: 0; 
    'umap__n_components': Integer(10, 100), #GEORGE 1, 2
    'gmm__n_components': Integer(3, 15),
    'gmm__covariance_type': Categorical(['full', 'diag', 'spherical']) # tied
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        help="Config file", required=True)
    parser.add_argument("--run_id", dest="run_id",
                        help="Experiment ID (seen in wandb)", required=True)
    return parser.parse_args()


def create_cluster_image(config, run_id):
   
    device = config["device"]
    DIR = f"{os.environ['FILESDIR']}/data/clustering/{run_id}"
    # the embeddings and the images are saved in the same order
    C_emb = torch.load(f"{DIR}/classifier_embeddings.pt")
    images = torch.load(f"{DIR}/images_acd_1.pt").to(device)

    config_run = {
        'clustering_method': 'gmm',
        'reduce_method': 'umap'
    }
    NAME = config_run['clustering_method']+'_'+config_run['reduce_method']

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-4-clustering_optimize_{NAME}',
                name=f"{run_id}",
                config=config_run)
    
    # get embeddings
    with torch.no_grad():
        embeddings = C_emb(images).cpu().detach().numpy()

    # Create GridSearchCV object with silhouette scoring 
    # TODO train test split?
    print("Starting optimization...")
    bayes_search = BayesSearchCV(PIPELINE, scoring=sil_score, search_spaces=PARAM_SPACE, cv=5, random_state=2, n_jobs=-1, verbose=1, n_iter=50)
    bayes_search.fit(embeddings)
    clustering_result = bayes_search.predict(embeddings)
    # get the embeddings reduced
    embeddings_red = bayes_search.best_estimator_['umap'].transform(embeddings)
    print(bayes_search.best_params_)

    # get prototypes of each cluster
    prototypes = [calculate_medoid(embeddings_red[clustering_result == cl_label]) for cl_label in np.unique(clustering_result) if cl_label >= 0]
    proto_idx = [np.where(np.all(embeddings_red == el, axis=1))[0][0] for el in prototypes]

    # calculate test embeddings and scores
    print("Calculating test embeddings...")
    embeddings_tst, preds = calculate_test_embeddings(config["dataset"]["name"], config["data-dir"], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], config['train']['step-2']['batch-size'], device, C_emb)

    print("Start reporting...")
    # create wandb report
    create_wandb_report_metrics(wandb, embeddings_red, clustering_result)
    create_wandb_report_images(wandb, NAME, images, clustering_result, proto_idx)
    create_wandb_report_2dviz(wandb, NAME, embeddings, embeddings_tst, preds, clustering_result, proto_idx)

    wandb.finish()


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # read configs
    config = read_config(args.config)
    create_cluster_image(config, args.run_id)
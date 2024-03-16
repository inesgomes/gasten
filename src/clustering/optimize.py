import os
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from src.clustering.aux import parse_args, get_clustering_path, calculate_medoid, sil_score, create_wandb_report_metrics, create_wandb_report_images, create_wandb_report_2dviz, calculate_test_embeddings
from src.utils.config import read_config_clustering
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from umap import UMAP
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# Define the pipeline with UMAP and GMM
PIPELINE = Pipeline(steps=[
    ('umap', UMAP()),
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

def hyper_tunning_clusters(config, classifier):

    # confirm that we are working with the expected algorithms
    if PIPELINE.steps[0][0] != config['clustering']['dim-reduction']:
        print("Error: please modify PIPELINE and PARAM_SPACE  to receive the expected dimensionality reduction method")
        exit(1)
    if PIPELINE.steps[1][0] != config['clustering']['clustering']:
        print("Error: please modify PIPELINE and PARAM_SPACE to receive the expected clustering method")
        exit(1)
   
    device = config["device"]
    classifier_name = classifier.split('/')[-1]
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    print(path)
    # the embeddings and the images are saved in the same order
    C_emb = torch.load(f"{path}/classifier_embeddings.pt")
    acd = int(config['clustering']['acd']*10)
    images = torch.load(f"{path}/images_acd_{acd}.pt").to(device)

    config_run = {
        'reduce_method': config['clustering']['dim-reduction'],
        'clustering_method': config['clustering']['clustering']
    }
    NAME = config_run['clustering_method']+'_'+config_run['reduce_method']

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-4-clustering_optimize_{NAME}',
                name=f"{config['gasten']['run-id']}",
                config=config_run)
    
    # get embeddings
    with torch.no_grad():
        embeddings_ori = C_emb(images)
        embeddings = embeddings_ori.detach().cpu().numpy()

    # Create GridSearchCV object with silhouette scoring 
    print("Starting optimization...")
    # TODO change n_iter=70
    bayes_search = BayesSearchCV(PIPELINE, scoring=sil_score, search_spaces=PARAM_SPACE, cv=5, random_state=2, n_jobs=-1, verbose=1, n_iter=2)
    bayes_search.fit(embeddings)
    clustering_result = bayes_search.predict(embeddings)
    # get the embeddings reduced
    embeddings_red = bayes_search.best_estimator_['umap'].transform(embeddings)
    print(bayes_search.best_params_)

    # get prototypes of each cluster
    prototypes = [calculate_medoid(embeddings_red[clustering_result == cl_label]) for cl_label in np.unique(clustering_result) if cl_label >= 0]
    proto_idx = [np.where(np.all(embeddings_red == el, axis=1))[0][0] for el in prototypes]
    proto_idx_torch = torch.tensor(proto_idx).to(device)

    # calculate test embeddings and scores
    print("Calculating test embeddings...")
    embeddings_tst, preds = calculate_test_embeddings(config["dataset"]["name"], config["dir"]['data'], config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"], config['batch-size'], device, C_emb)

    print("Start reporting...")
    # create wandb report
    create_wandb_report_metrics(wandb, embeddings_red, clustering_result)
    create_wandb_report_images(wandb, NAME, images, clustering_result, proto_idx_torch)
    create_wandb_report_2dviz(wandb, NAME, embeddings_ori, embeddings_tst, proto_idx_torch, preds, clustering_result)

    wandb.finish()


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # read configs
    config = read_config_clustering(args.config)
    hyper_tunning_clusters(config, config['gasten']['classifier'][0])

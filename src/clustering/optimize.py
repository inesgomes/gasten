import os
import torch
import wandb
from dotenv import load_dotenv
from src.clustering.aux import get_clustering_path, sil_score, create_wandb_report_metrics, parse_args
from src.utils.config import read_config_clustering
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from umap import UMAP


METHODS = {
    'umap': UMAP(metric='cosine'),
    'gmm': GaussianMixture(random_state=2, covariance_type='full', init_params='k-means++'), # full -> N2D
    'hdbscan': HDBSCAN(cluster_selection_method='leaf')
}

PARAM_SPACE = {
    'umap': {
        'umap__n_neighbors': Integer(5, 25), #N2D: 20
        'umap__min_dist': Real(0.01, 0.25), #N2D: 0; 
        'umap__n_components': Integer(5, 80), #GEORGE 1, 2
    },
    'gmm': {
        'gmm__n_components': Integer(2, 10)
    },
    'hdbscan': {
        'hdbscan__cluster_selection_epsilon': Real(0, 5),
        'hdbscan__min_cluster_size': Integer(3, 10),
    }
}

def load_gasten_images(config, classifier_name):
    """
    """
    print("> Load previous step ...")
    # classifier
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    C_emb = torch.load(f"{path}/classifier_embeddings.pt")
    # images
    acd = int(config['clustering']['acd']*10)
    images = torch.load(f"{path}/images_acd_{acd}.pt")
    # get embeddings
    with torch.no_grad():
        embeddings_ori = C_emb(images)

    return C_emb, images, embeddings_ori

def save_estimator(config, estimator, classifier_name, estimator_name):
    """
    """
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    torch.save(estimator, f"{path}/{estimator_name}.pt")
    
def hyper_tunning_clusters(config, classifier_name, dim_reduction, clustering, embeddings_ori):
   
    config_run = {
        'reduce_method': dim_reduction,
        'clustering_method': clustering
    }
    estimator_name = config_run['clustering_method']+'_'+config_run['reduce_method']

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-4-clustering_optimize_{estimator_name}',
                name=f"{config['gasten']['run-id']}-{classifier_name}_v2",
                config=config_run)
    
    pipeline = Pipeline(steps=[
        (dim_reduction, METHODS[dim_reduction]),
        (clustering, METHODS[clustering])
    ])
    
    param_space = {**PARAM_SPACE[dim_reduction], **PARAM_SPACE[clustering]}
    embeddings = embeddings_ori.detach().cpu().numpy()

    # Create GridSearchCV object with silhouette scoring 
    print("> Starting optimization ...")
    bayes_search = BayesSearchCV(pipeline, scoring=sil_score, search_spaces=param_space, cv=5, random_state=2, n_jobs=-1, verbose=1, n_iter=config["clustering"]["n-iter"])
    bayes_search.fit(embeddings)
    clustering_result = bayes_search.predict(embeddings)
    # get the embeddings reduced
    embeddings_red = bayes_search.best_estimator_[dim_reduction].transform(embeddings)

    print("> Start reporting...")
    # save best paramters
    wandb.log(bayes_search.best_params_)
    create_wandb_report_metrics(embeddings_red, clustering_result)

    wandb.finish()
    return bayes_search.best_estimator_, embeddings_red, clustering_result

if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # read configs
    config = read_config_clustering(args.config)
    for clf in config['gasten']['classifier']:
        classifier_name = clf.split('/')[-1]
        _, _, embeddings_ori = load_gasten_images(config, classifier_name)
        for opt in config['clustering']['options']:
            estimator, _, _ = hyper_tunning_clusters(config, classifier_name, opt['dim-reduction'], opt['clustering'], embeddings_ori)
            if config["checkpoint"]:
                save_estimator(config, estimator, classifier_name, f"{opt['dim-reduction']}_{opt['clustering']}")

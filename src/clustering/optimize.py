import os
import torch
import wandb
from dotenv import load_dotenv
from src.clustering.generate_embeddings import load_gasten
from src.clustering.aux import get_clustering_path, parse_args
from src.clustering.visualizations import create_wandb_report_metrics
from src.utils.config import read_config_clustering
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from umap import UMAP


METHODS = {
    'umap': UMAP(metric='cosine'),
    'gmm': GaussianMixture(random_state=2, covariance_type='full', init_params='k-means++'), # full -> N2D
    'tsne': TSNE(random_state=2),
    'hdbscan': HDBSCAN(cluster_selection_method='leaf', store_centers="medoid", allow_single_cluster=True, min_samples=5)
}

PARAM_SPACE = {
    'umap': {
        'umap__n_neighbors': Integer(5, 25), #N2D: 20
        'umap__min_dist': Real(0.01, 0.25), #N2D: 0; 
        'umap__n_components': Integer(5, 80), #GEORGE 1, 2
    },
    'tsne': {
        'tsne__perplexity': Integer(5, 30),
    },
    'gmm': {
        'gmm__n_components': Integer(2, 10)
    },
    'hdbscan': {
        'hdbscan__cluster_selection_epsilon': Real(0, 5),
        'hdbscan__min_cluster_size': Integer(3, 10),
    }
}

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

def load_gasten_images(config, C_emb, classifier_name):
    """
    """
    print("> Load previous step ...")
    # classifier
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    # images
    acd = int(config['clustering']['acd']*10)
    syn_images_filtered = torch.load(f"{path}/images_acd_{acd}.pt")
    # get embeddings
    with torch.no_grad():
        syn_embeddings_filt = C_emb(syn_images_filtered)
    return syn_images_filtered, syn_embeddings_filt

def save_estimator(config, estimator, classifier_name, estimator_name):
    """
    """
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    torch.save(estimator, f"{path}/{estimator_name}.pt")
    
def hyper_tunning_clusters(config, classifier_name, dim_reduction, clustering, syn_embeddings_f):
   
    config_run = {
        'step': 'clustering_optimization',
        'classifier_name': classifier_name,
        'estimator_name': dim_reduction+'_'+clustering,
    }

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type=f'step-4-clustering_optimize_{config_run['estimator_name']}',
                name=f"{config['gasten']['run-id']}-{classifier_name}_{config['tag']}",
                tags=[config["tag"]],
                config=config_run)
    
    pipeline = Pipeline(steps=[
        (dim_reduction, METHODS[dim_reduction]),
        (clustering, METHODS[clustering])
    ])
    param_space = {**PARAM_SPACE[dim_reduction], **PARAM_SPACE[clustering]}
    embeddings = syn_embeddings_f.detach().cpu().numpy()

    # Create GridSearchCV object with silhouette scoring 
    print("> Starting optimization ...")
    # TODO: this step is very slow -> optimize
    bayes_search = BayesSearchCV(pipeline, scoring=sil_score, search_spaces=param_space, cv=5, random_state=2, n_jobs=-1, verbose=1, n_iter=config["clustering"]["n-iter"])
    bayes_search.fit(embeddings)

    embeddings_red = bayes_search.best_estimator_[0].fit_transform(embeddings)
    clustering_result = bayes_search.best_estimator_[1].fit_predict(embeddings_red)
    # get the embeddings reduced

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
        _, _, C_emb, classifier_name = load_gasten(config, clf)
        _, syn_embeddings_f = load_gasten_images(config, C_emb, classifier_name)
        for opt in config['clustering']['options']:
            estimator, _, _ = hyper_tunning_clusters(config, classifier_name, opt['dim-reduction'], opt['clustering'], syn_embeddings_f)
            if config["checkpoint"]:
                save_estimator(config, estimator, classifier_name, f"{opt['dim-reduction']}_{opt['clustering']}")

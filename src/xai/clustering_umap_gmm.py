import os
import argparse
from datetime import datetime
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from src.metrics import fid
from src.utils.config import read_config
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from umap import UMAP
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.spatial.distance import cdist
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader



def get_gasten_info(config):
    classifier_name = config['train']['step-2']['classifier'][0].split(
        '/')[-1].split('.')[0]
    weight = config['train']['step-2']['weight'][0]
    epoch1 = config['train']['step-2']['step-1-epochs'][0]
    return classifier_name, weight, epoch1


def get_gan_path(config, run_id, epoch2):
    project = config['project']
    name = config['name']
    classifier_name, weight, epoch1 = get_gasten_info(config)

    # find directory whose name ends with a given id
    for dir in os.listdir(f"{os.environ['FILESDIR']}/out/{config['project']}/{config['name']}"):
        if dir.endswith(run_id):
            return f"{os.environ['FILESDIR']}/out/{project}/{name}/{dir}/{classifier_name}_{weight}_{epoch1}/{epoch2}"

    raise Exception(f"Could not find directory with id {run_id}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        help="Config file", required=True)
    parser.add_argument("--run_id", dest="run_id",
                        help="Experiment ID (seen in wandb)", required=True)
    parser.add_argument("--epoch", dest="gasten_epoch", required=True)
    parser.add_argument("--acd_threshold", dest="acd_threshold", default=0.1)
    return parser.parse_args()
    

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


def find_closest_point(target_point, dataset):
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
    # Calculate pairwise distances
    distances = cdist(cluster_points, cluster_points, metric='euclidean')

    # Find the index of the point with the smallest sum of distances
    medoid_index = np.argmin(np.sum(distances, axis=0))

    # Retrieve the medoid point
    medoid = cluster_points[medoid_index]

    return medoid


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    print(estimator)
    return -estimator['gmm'].bic(X)

def sil_score(estimator, X):
    """_summary_

    Args:
        estimator (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_red = estimator['umap'].fit_transform(X)
    labels = estimator['gmm'].fit_predict(x_red)
    return silhouette_score(x_red, labels)


def create_cluster_image():
    # setup
    load_dotenv()
    args = parse_args()

    SKIP_FID_SCORE = True

    # read configs
    config = read_config(args.config)
    device = config["device"]
    batch_size = config['train']['step-2']['batch-size']
    n_images = config['fixed-noise']
    # prepare wandb info
    dataset_id = datetime.now().strftime("%b%dT%H-%M")
    classifier_name, weight, epoch1 = get_gasten_info(config)    
    config_run = {
        'classifier': classifier_name,
        'gasten': {
            'weight': weight,
            'epoch1': epoch1,
            'epoch2': args.gasten_epoch,
        },
        'probabilities': {
            'min': 0.5 - args.acd_threshold,
            'max': 0.5 + args.acd_threshold
        }
    }

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type='step-3-amb_img_generation',
                name=dataset_id,
                config=config_run)

    # get GAN
    gan_path = get_gan_path(
        config, args.run_id, config_run['gasten']['epoch2'])
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)

    # get classifier
    C, _, _, _ = construct_classifier_from_checkpoint(
        config['train']['step-2']['classifier'][0], device=device)

    # remove last layer of classifier to get the embeddings
    C_emb = torch.nn.Sequential(*list(C.children())[0][:-1])

    # prepare FID calculation
    if not SKIP_FID_SCORE:
        mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
        fm_fn, dims = fid.get_inception_feature_map_fn(device)
        fid_metric = fid.FID(fm_fn, dims, n_images, mu, sigma, device=device)

    # create fake images
    test_noise = torch.randn(n_images, config["model"]["z_dim"], device=device)
    noise_loader = DataLoader(TensorDataset(test_noise), batch_size=batch_size, shuffle=False)
    images_array = []
    desc = 'Generating fake images' if SKIP_FID_SCORE else 'Evaluating fake images'
    for idx, batch in enumerate(tqdm(noise_loader, desc=desc)):
        with torch.no_grad():
            netG.eval()
            batch_images = netG(*batch)
        images_array.append(batch_images)
        
        # calculate FID score - all images
        if not SKIP_FID_SCORE:
            max_size = min(idx*batch_size, n_images)
            fid_metric.update(batch_images, (idx*batch_size, max_size))
        

    # FID for fake images
    if not SKIP_FID_SCORE:
        wandb.log({"fid_score_all": fid_metric.finalize()})
        fid_metric.reset()

    # Concatenate batches into a single array
    images = torch.cat(images_array, dim=0)

    # apply classifier to fake images
    with torch.no_grad():
        C.to(device)
        C.eval()
        pred = C(images)

    # filter images so that ACD < 0.1
    mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
    images_mask = images[mask]

    # point to the original positions (needed later for viz)
    original_pos = np.where(mask.cpu().detach().numpy())[0]

    # count the ambig images
    n_amb_img = images_mask.shape[0]
    wandb.log({"n_ambiguous_images": n_amb_img})

    # calculate FID score in batches - ambiguous images
    if not SKIP_FID_SCORE:
        image_loader = DataLoader(TensorDataset(images_mask), batch_size=batch_size, shuffle=False)
        for idx, batch in enumerate(tqdm(image_loader, desc='Evaluating ambiguous fake images')):
            max_size = min(idx*batch_size, n_images)
            fid_metric.update(*batch, (idx*batch_size, max_size))
        
        wandb.log({"fid_score_ambiguous": fid_metric.finalize()})
        fid_metric.reset()

    # get the embeddings for the ambiguous images
    with torch.no_grad():
        C_emb.to(device)
        C_emb.eval()
        embeddings_f = C_emb(images_mask).cpu().detach().numpy()
   
    # close wandb
    wandb.finish()

    config_run['clustering_method'] = 'gmm'
    config_run['reduce_method'] = 'umap'
    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type='step-4-clustering_optimize_gmm_umap',
                name=f"{dataset_id}_v2",
                config=config_run)

    # Define the pipeline with UMAP and GMM
    pipeline = Pipeline(steps=[
        ('umap', UMAP()),
        ('gmm', GaussianMixture(random_state=2)) # full -> N2D
    ])
    # Define the parameter grid for UMAP and GMM
    param_space = {
        'umap__n_neighbors': Integer(5, 25), #N2D: 20
        'umap__min_dist': Real(0.01, 0.5), #N2D: 0; 
        'umap__n_components': Integer(10, 100), #GEORGE 1, 2
        'gmm__n_components': Integer(3, 15),
        'gmm__covariance_type': Categorical(['full', 'diag', 'spherical']) # tied
    }

    # Create GridSearchCV object with silhouette scoring 
    bayes_search = BayesSearchCV(pipeline, scoring=sil_score, search_spaces=param_space, cv=5, random_state=8, n_jobs=6, verbose=1, n_iter=80)
    bayes_search.fit(embeddings_f)
    clustering_result = bayes_search.predict(embeddings_f)
    # get the embeddings reduced
    embeddings_red = bayes_search.best_estimator_['umap'].transform(embeddings_f)
    print(bayes_search.best_params_)

    # evaluate the clustering
    wandb.log({"silhouette_score": silhouette_score(embeddings_red, clustering_result)})
    wandb.log({"calinski_harabasz_score": calinski_harabasz_score(embeddings_red, clustering_result)})
    wandb.log({"davies_bouldin_score": davies_bouldin_score(embeddings_red, clustering_result)})
    # log cluster information
    wandb.log({"n_clusters": np.unique(clustering_result)})
    cluster_sizes = pd.Series(clustering_result).value_counts().reset_index()
    wandb.log({"cluster_sizes": wandb.Table(dataframe=cluster_sizes)})

    # save images per cluster
    for cl_label, cl_examples in pd.DataFrame({'cluster': clustering_result, 'original_pos': original_pos}).groupby('cluster'):
        # get the original image positions
        mask = np.full(n_images, False)
        mask[cl_examples['original_pos']] = True
        wandb.log({"cluster_images": wandb.Image(images[mask], caption=f"gmm_umap | Label {cl_label} | (N = {cl_examples.shape[0]})")})

    # get prototypes of each cluster
    prototypes = [calculate_medoid(embeddings_red[clustering_result == cl_label]) for cl_label in np.unique(clustering_result) if cl_label >= 0]
    proto_idx = [np.where(np.all(embeddings_red == el, axis=1))[0][0] for el in prototypes]

    # save images
    mask = np.full(n_images, False)
    mask[original_pos[proto_idx]] = True
    wandb.log({"prototypes": wandb.Image(images[mask], caption="gmm_umap")})

    wandb.finish()


if __name__ == "__main__":
    create_cluster_image()
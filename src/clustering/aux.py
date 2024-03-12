
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_gasten_info(config):
    """_summary_

    Args:
        config (_type_): _description_

    Returns:
        _type_: _description_
    """
    classifier_name = config['train']['step-2']['classifier'][0].split(
        '/')[-1].split('.')[0]
    weight = config['train']['step-2']['weight'][0]
    epoch1 = config['train']['step-2']['step-1-epochs'][0]
    return classifier_name, weight, epoch1


def get_gan_path(config, run_id, epoch2):
    """_summary_

    Args:
        config (_type_): _description_
        run_id (_type_): _description_
        epoch2 (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    project = config['project']
    name = config['name']
    classifier_name, weight, epoch1 = get_gasten_info(config)

    # find directory whose name ends with a given id
    for dir in os.listdir(f"{os.environ['FILESDIR']}/out/{config['project']}/{config['name']}"):
        if dir.endswith(run_id):
            return f"{os.environ['FILESDIR']}/out/{project}/{name}/{dir}/{classifier_name}_{weight}_{epoch1}/{epoch2}"

    raise Exception(f"Could not find directory with id {run_id}")
    

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

def db_score(estimator, X):
    """_summary_

    Args:
        estimator (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_red = estimator['umap'].fit_transform(X)
    labels = estimator['gmm'].fit_predict(x_red)
    return -davies_bouldin_score(x_red, labels)
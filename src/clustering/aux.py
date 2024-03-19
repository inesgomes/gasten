import argparse
import os
import torch
from src.datasets import load_dataset
from sklearn.metrics import silhouette_score, davies_bouldin_score


def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        required=True, help="Config file from experiments/clustering folder")
    return parser.parse_args()

def get_clustering_path(clustering_path, run_id, classifier):
    path = f"{clustering_path}/{run_id}/{classifier}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_gan_path(config, classifier_name):
    """_summary_

    Args:
        config (_type_): _description_
        classifier (_str_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    run_id = config["gasten"]["run-id"]
    project = config["project"]
    name = config["name"]
    # find directory whose name ends with a given id
    for dir in os.listdir(f"{os.environ['FILESDIR']}/out/{project}/{config['name']}"):
        if dir.endswith(run_id):
            return f"{os.environ['FILESDIR']}/out/{project}/{name}/{dir}/{classifier_name.split(".")[0]}_{config['gasten']['weight']}_{config['gasten']['epoch']['step-1']}/{config['gasten']['epoch']['step-2']}"
    raise Exception(f"Could not find directory with id {run_id}")

def calculate_test_embeddings(dataset_name, data_dir, pos, neg, batch_size, device, C_emb):
    test_set = load_dataset(dataset_name, data_dir, pos, neg, train=False)[0]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)
    embeddings_tst_array = []
    y_test = []
    with torch.no_grad():
        for data_tst in test_loader:
            X, y = data_tst
            embeddings_tst_array.append(C_emb(X.to(device)))
            y_test.append(y)

    # concatenate the array
    embeddings_tst = torch.cat(embeddings_tst_array, dim=0)
    y_test = torch.cat(y_test, dim=0).cpu().detach().numpy()
    return embeddings_tst, y_test

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

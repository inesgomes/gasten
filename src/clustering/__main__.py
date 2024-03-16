from dotenv import load_dotenv
from src.clustering.aux import parse_args
from src.utils.config import read_config_clustering
from src.clustering.generate_embeddings import generate_embeddings
from src.clustering.optimize import hyper_tunning_clusters


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # read configs
    config = read_config_clustering(args.config)

    for classifier in config['gasten']['classifier']:
        generate_embeddings(config, classifier)
        hyper_tunning_clusters(config, classifier)

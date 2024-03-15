from dotenv import load_dotenv
from src.clustering.aux import parse_args
from src.utils.config import read_config_clustering

if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # read configs
    config = read_config_clustering(args.config)
    # generate ambiguous images
    generate_embeddings(config)
    # TODO: create clusters
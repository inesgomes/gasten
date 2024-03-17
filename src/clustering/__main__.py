from dotenv import load_dotenv
from src.clustering.aux import parse_args
from src.utils.config import read_config_clustering
from src.clustering.generate_embeddings import generate_embeddings, load_gasten, save_gasten_images
from src.clustering.optimize import hyper_tunning_clusters, save_estimator
from src.clustering.prototypes import calculate_prototypes


def save(config, C_emb, images, estimator, classifier_name, estimator_name):
    """
    """
    print("> Save ...")
    save_gasten_images(config, C_emb, images, classifier_name)
    save_estimator(config, estimator, classifier_name, estimator_name)


def baseline(config, classifier_name, n_samples=10):
    # load test set
    # filter by ACD
    # select random n_samples
    # evaluate - same as prototypes
    # - check images
    # - check embeddings visualization
    # - check metrics
    pass


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # read configs
    config = read_config_clustering(args.config)

    for clf in config['gasten']['classifier']:
        # generate images
        netG, C, classifier_name = load_gasten(config, clf)
        C_emb, images, embeddings_ori = generate_embeddings(config, netG, C, classifier_name)
        for opt in config['clustering']['options']:
            # apply clustering
            estimator, clustering_result, embeddings_reduced = hyper_tunning_clusters(config, classifier_name, opt['dim-reduction'], opt['clustering'], embeddings_ori)
            estimator_name = f"{opt['dim-reduction']}_{opt['clustering']}"
            # get prototypes
            #for typ in config['prototypes']['type']: 
                #calculate_prototypes(config, typ, classifier_name, estimator_name, images, embeddings_ori, embeddings_reduced, clustering_result)

            if config["checkpoint"]:
                save(config, C_emb, images, estimator, classifier_name, estimator_name)

        # TODO calculate the classifier baseline: select from test set images with prob < 0.1 acd (randomly 10?)


from dotenv import load_dotenv
from src.clustering.aux import parse_args
from src.utils.config import read_config_clustering
from src.clustering.generate_embeddings import generate_embeddings, load_gasten, save_gasten_images
from src.clustering.optimize import save_estimator, hyper_tunning_clusters
from src.clustering.prototypes import baseline_prototypes, calculate_prototypes


def save(config, C_emb, images, estimator, classifier_name, estimator_name):
    """
    """
    print("> Save ...")
    save_gasten_images(config, C_emb, images, classifier_name)
    save_estimator(config, estimator, classifier_name, estimator_name)
   

if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    # read configs
    config = read_config_clustering(args.config)

    for clf in config['gasten']['classifier']:
        # load previous step
        netG, C, C_emb, classifier_name = load_gasten(config, clf)
        # calculate baseline
        baseline_prototypes(config, classifier_name, C, C_emb, 5, iter=0)
        # generate images
        syn_images_f, syn_embeddings_f = generate_embeddings(config, netG, C, C_emb, classifier_name)
        # calculate prototypes via clustering
        for opt in config['clustering']['options']:
            # apply clustering
            estimator, embeddings_reduced, clustering_result = hyper_tunning_clusters(config, classifier_name, opt['dim-reduction'], opt['clustering'], syn_embeddings_f)
            estimator_name = f"{opt['dim-reduction']}_{opt['clustering']}"
            # get prototypes
            for typ in config['prototypes']['type']:
                calculate_prototypes(config, typ, classifier_name, estimator_name, C, C_emb, syn_images_f, syn_embeddings_f, embeddings_reduced, clustering_result)

            if config["checkpoint"]:
                save(config, C_emb, syn_images_f, estimator, classifier_name, estimator_name)
                
        # TODO: sensitivity analysis - no visualizations
        # after having the hyperparamter tunning, re-run the image generation and apply the best clustering, 5 times
        # extract the prototypes and calculate cosine similarity of the original embeddings
        # hypothesis is that the prototypes are similar between them but different from the baseline (all-vs-all)


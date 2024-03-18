import os
from dotenv import load_dotenv
from src.utils.config import read_config_clustering
from src.clustering.aux import get_gan_path, parse_args, get_clustering_path
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint
from src.metrics import fid
import wandb
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


def load_gasten(config, classifier):
    """
    load information from previous step
    """
    device = config["device"]
    # get classifier name
    classifier_name = classifier.split("/")[-1]
    # get GAN
    gan_path = get_gan_path(config, classifier_name)
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    # get classifier
    C, _, _, _ = construct_classifier_from_checkpoint(classifier, device=device)

    return netG, C, classifier_name


def save_gasten_images(config, classifier, images, classifier_name):
    """
    save embeddings and images for next step
    """
    path = get_clustering_path(config['dir']['clustering'], config['gasten']['run-id'], classifier_name)
    torch.save(classifier, f"{path}/classifier_embeddings.pt")
    thr = int(config['clustering']['acd']*10)
    torch.save(images, f"{path}/images_acd_{thr}.pt")


def generate_embeddings(config, netG, C, classifier_name):

    device = config["device"]
    batch_size = config['batch-size']

    config_run = {
        'gasten': {
            'epoch1': config['gasten']['epoch']['step-1'],
            'epoch2': config['gasten']['epoch']['step-2'],
            'weight': config['gasten']['weight']
        },
        'probabilities': {
            'min': 0.5 - config['clustering']['acd'],
            'max': 0.5 + config['clustering']['acd']
        },
        'generated_images': config['clustering']['fixed-noise']
    }

    wandb.init(project=config['project'],
                dir=os.environ['FILESDIR'],
                group=config['name'],
                entity=os.environ['ENTITY'],
                job_type='step-3-amb_img_generation',
                name=f"{config['gasten']['run-id']}-{classifier_name}_v2",
                config=config_run)

    # remove last layer of classifier to get the embeddings
    C.eval()
    C_emb = torch.nn.Sequential(*list(C.children())[0][:-1])
    C_emb.eval()

    # prepare FID calculation
    if config['compute-fid']:
        mu, sigma = fid.load_statistics_from_path(config['dir']['fid-stats'])
        fm_fn, dims = fid.get_inception_feature_map_fn(device)
        fid_metric = fid.FID(fm_fn, dims, config_run['generated_images'], mu, sigma, device=device)

    # create fake images
    test_noise = torch.randn(config_run['generated_images'], config["clustering"]["z-dim"], device=device)
    noise_loader = DataLoader(TensorDataset(test_noise), batch_size=batch_size, shuffle=False)
    images_array = []
    for idx, batch in enumerate(tqdm(noise_loader, desc='Evaluating fake images')):
        # generate images
        with torch.no_grad():
            netG.eval()
            batch_images = netG(*batch)
        
        # calculate FID score - all images
        if config['compute-fid']:
            max_size = min(idx*batch_size, config_run['generated_images'])
            fid_metric.update(batch_images, (idx*batch_size, max_size))
            # FID for fake images
            wandb.log({"fid_score_all": fid_metric.finalize()})
            fid_metric.reset()

        images_array.append(batch_images)

    # Concatenate batches into a single array
    images = torch.cat(images_array, dim=0)

    # apply classifier to fake images
    with torch.no_grad():
        pred = C(images).cpu().detach().numpy()

    # filter images so that ACD < threshold
    mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
    images_mask = images[mask]

    # count the ambig images
    n_amb_img = images_mask.shape[0]
    wandb.log({"n_ambiguous_images": n_amb_img})

    # calculate FID score in batches - ambiguous images
    if config['compute-fid']:
        image_loader = DataLoader(TensorDataset(images_mask), batch_size=batch_size, shuffle=False)
        for idx, batch in enumerate(tqdm(image_loader, desc='Evaluating ambiguous fake images')):
            max_size = min(idx*batch_size, config_run['generated_images'])
            fid_metric.update(*batch, (idx*batch_size, max_size))
    
        wandb.log({"fid_score_ambiguous": fid_metric.finalize()})
        fid_metric.reset()

    # get embeddings
    with torch.no_grad():
        embeddings_ori = C_emb(images)

    #visualize_embeddings(config, C_emb, pred[mask, embeddings_ori)

    # close wandb
    wandb.finish()

    return C_emb, images_mask, embeddings_ori


if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()
    config = read_config_clustering(args.config)
    for classifier in config['gasten']['classifier']:
        netG, C, classifier_name = load_gasten(config, classifier)
        C_emb, images, _ = generate_embeddings(config, netG, C, classifier_name)
        if config["checkpoint"]:
            save_gasten_images(config, C_emb, images, classifier_name)

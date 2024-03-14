from email.mime import image
import os
import torch
import wandb
from dotenv import load_dotenv
from src.utils.config import read_config
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint
import argparse
from omnixai.explainers.vision import CounterfactualExplainer
import torchvision
from omnixai.data.image import Image


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
    return f"{os.environ['FILESDIR']}/out/{project}/{name}/{run_id}/{classifier_name}_{weight}_{epoch1}/{epoch2}"


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

if __name__ == "__main__":
    # setup
    load_dotenv()
    args = parse_args()

    # read configs
    config = read_config("experiments/mnist_7v1_1iter.yml")
    device = torch.device("cpu") #config["device"])
    n_images = config['fixed-noise']
    pos_class = config["dataset"]["binary"]["pos"]
    neg_class = config["dataset"]["binary"]["neg"]

    classifier_name, weight, epoch1 = get_gasten_info(config)
    config_run = {
        'id': "Jul10T16-31_3l4yezei",
        'classifier': classifier_name,
        'gasten': {
            'weight': weight,
            'epoch1': epoch1,
            'epoch2': 40,
        },
        'probabilities': {
            'min': 0.3,
            'max': 0.5,
        }
    }
    
    # start wandb
    #wandb.init(project=config['project'],
    #           dir=os.environ['FILESDIR'],
    #           group=config['name'],
    #           entity=os.environ['ENTITY'],
    #           job_type='xai',
    #           name=f"{config_run['classifier']}-counterfactuals",
    #           config=config_run)
    
    # get GAN
    gan_path = get_gan_path(
        config, config_run['id'], config_run['gasten']['epoch2'])
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    fixed_noise = torch.randn(
        n_images, config["model"]["z_dim"], device=device)

    # get classifier
    net, _, _, _ = construct_classifier_from_checkpoint(
        config['train']['step-2']['classifier'][0], device=device)

    # create fake images and apply classifier
    with torch.no_grad():
        netG.eval()
        images = netG(fixed_noise)
        net.eval()
        pred = net(images)

    # remove images with high probability
    mask = (pred >= config_run['probabilities']['min']) & (pred <= config_run['probabilities']['max'])
    images_mask = images[mask][0]

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               ])
    preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])
    images_pil = Image(images_mask.cpu().detach().numpy(), batched=True, channel_last=False)

    explainer = CounterfactualExplainer(
        model=net,
        preprocess_function=preprocess,
        c=10.0,
        kappa=10.0,
        binary_search_steps=20,
        learning_rate=1e-2,
        num_iterations=100,
        grad_clip=1e3,
    )
    
    explanations = explainer.explain(images_pil)
    fig = explanations.ipython_plot(index=0)

    # save image
    #fig.savefig(f"{os.environ['FILESDIR']}/out/{config['project']}/{config['name']}/{config_run['id']}/counterfactuals.png")

    #wandb.log({f"counterfactual": wandb.Image(fig)})

    # close wandb
    #wandb.finish()
"""_summary_
generate images with GASTeN to first check the data
Some relevant info:
> Jul10T16-19_1eygesna -> z=200
> Jul10T16-31_3l4yezei -> z=2000

- step1 GAN: /media/SEAGATE6T/IGOMES/gasten/out/gasten_20230710/mnist-7v1/Jul10T17-29_1p7i0mep/step-1/10 
- step2 GAN: /media/SEAGATE6T/IGOMES/gasten/out/gasten_20230710/mnist-7v1/Jul10T17-29_1p7i0mep/cnn-2-1_25_10/40
"""
import os
import torch
import wandb
from torchvision.utils import save_image
from dotenv import load_dotenv
from src.utils.config import read_config
from src.utils.checkpoint import construct_classifier_from_checkpoint, construct_gan_from_checkpoint


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


def get_experiment_number():
    no = 1
    path = f"{os.environ['FILESDIR']}/data/gasten"
    files = os.listdir(path)
    if len(files) > 0:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
        no = int(files[-1].split("_")[-1].split(".")[0]) + 1
    return no


if __name__ == "__main__":
    ### 
    # SETUP
    ###
    load_dotenv()
    config = read_config("experiments/mnist_7v1_1iter.yml")

    device = torch.device(config["device"])
    classifier_name, weight, epoch1 = get_gasten_info(config)
    gasten_name = get_experiment_number()
    config_run = {
        'id': "Jul10T16-31_3l4yezei",
        'name': gasten_name,
        'min_prob': 0.45,
        'max_prob': 0.55,
        'classifier': classifier_name,
        'gasten': {
            'weight': weight,
            'epoch1': epoch1,
            'epoch2': 40,
        },
        'image_path' : f"{os.environ['FILESDIR']}/data/gasten/sample_{config['dataset']['binary']['neg']}vs{config['dataset']['binary']['pos']}_{gasten_name}"
    }

    # start experiment
    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='test-gasten',
               name=f"{config_run['id'].split('_')[-1]}_no-{config_run['name']}",
               config=config_run)

    # get GAN
    gan_path = get_gan_path(
        config, config_run['id'], config_run['gasten']['epoch2'])
    netG, _, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    fixed_noise = torch.randn(
        config['fixed-noise'], config["model"]["z_dim"], device=device)

    # get classifier
    net, _, _, _ = construct_classifier_from_checkpoint(config['train']['step-2']['classifier'][0], device=device)
    
    # create fake images and apply classifier
    with torch.no_grad():
        netG.eval()
        images = netG(fixed_noise)
        net.eval()
        pred = net(images)

    # select the fake images with prob between 0.45 and 0.55
    images_sel = images[(pred >= config_run['min_prob']) &
                        (pred <= config_run['max_prob'])]

    # save images (locally)
    #save_image(images_sel, f"{config_run['image_path']}.png", nrow=10)
    torch.save(images_sel, f"{config_run['image_path']}.pt")
    # save images (wandb)
    wandb.log({"image": wandb.Image(images_sel)})

    print(
        f"> Saved {images_sel.shape[0]} GASTeN images with probs between to {config_run['min_prob']} and {config_run['max_prob']} to {config_run['image_path']}")

    wandb.finish()

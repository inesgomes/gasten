import os
import random
import torch
from torchvision.utils import save_image
from dotenv import load_dotenv
from src.utils.config import read_config
from src.utils.checkpoint import construct_classifier_from_checkpoint


if __name__ == "__main__":
    """_summary_
    gerenate images with GASTeN to first check the data
    > Jul10T16-19_1eygesna -> z=200
    > Jul10T16-31_3l4yezei -> z=2000
    """
    load_dotenv()   

    # probs
    min_prob = 0.45
    max_prob = 0.55

    # SETUP
    config = read_config("experiments/mnist_7v1_1iter.yml")
    device = torch.device(config["device"])
    classifier = config['train']['step-2']['classifier'][0]

    # TODO use the experiment number and proper function to get the discriminator
    netG = torch.load(f"{os.environ['FILESDIR']}/out/test_generator.pt")
    fixed_noise = torch.randn(config['fixed-noise'], config["model"]["z_dim"], device=device)
    images = netG(fixed_noise)#.detach().cpu()

    # get classifier
    net, _, _, _ = construct_classifier_from_checkpoint(classifier, device=device)
    net.eval() 

    # apply classifier and select images with prob between 0.45 and 0.55
    pred = net(images)
    images_sel = images[(pred>=min_prob)&(pred<=max_prob)]

    # save images
    rnd = random.randint(1, 1000)
    path = f"{os.environ['FILESDIR']}/data/gasten/sample_{config['dataset']['binary']['neg']}vs{config['dataset']['binary']['pos']}_{rnd}"
    save_image(images_sel, path + ".png")
    torch.save(images_sel, f"{path}.pt")
    print(f"Saved {images_sel.shape[0]} GASTeN images with probs between to {min_prob} and {max_prob} to {path}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import random
from dotenv import load_dotenv
import os
from src.utils.checkpoint import construct_classifier_from_checkpoint
import argparse
from src.utils.config import read_config
import wandb
from src.datasets import load_dataset


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))
    
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, device)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

def train(autoencoder, data, device, epochs=20,):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device) 
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder

def interpolate(autoencoder, x_1, x_2, n=10):
    # create interpolations
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    return interpolate_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", help="Config file", default='experiments/mnist_7v1_1iter.yml')
    parser.add_argument("--train", dest="train", help="Train VAE", type=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    ##
    # SETUP
    ##
    load_dotenv()
    args = parse_args()
    config = read_config(args.config_path)

    device = torch.device(config["device"])
    pos_class = config["dataset"]["binary"]["pos"]
    neg_class = config["dataset"]["binary"]["neg"]

    config_run={
        'classifier': config['train']['step-2']['classifier'][0].split("/")[-1],
        'n_interpolations': 10,
        'latent_dims': 2,
        'vae_path': f"{os.environ['FILESDIR']}/models/vae/vae_{neg_class}vs{pos_class}.pt"
    }

    # get name
    path = f"{os.environ['FILESDIR']}/data/vae/"
    files = os.listdir(path)
    name = 1
    if len(files) > 0:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
        name = int(files[-1].split("_")[-1].split(".")[0]) + 1

    # start experiment
    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='vae_interpolation',
               name=f"{config_run['classifier']}-no-{name}",
               config=config_run)

    # LOAD data
    print(" > loading data ...") 
    dataset, _, _ = load_dataset(config["dataset"]["name"], config["data-dir"], pos_class, neg_class)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # train VAE
    if args.train is False:
        vae = torch.load(config_run['vae_path'])
    else:
        print(" > training VAE ...")
        vae = VariationalAutoencoder(config_run['latent_dims'], device).to(device)
        vae = train(vae, data_loader, device)
        torch.save(vae, config_run['vae_path'])

    print(" > creating interpolations ...")
    x, y = next(data_loader.__iter__())
    x_1 = x[y == 0][0].to(device)
    x_2 = x[y == 1][0].to(device)

    interpolate_list = interpolate(vae, x_1, x_2, config_run['n_interpolations'])

    # get classifier and visualize probabilities
    print(" > calculating probabilities ...")
    net, _, _, _ = construct_classifier_from_checkpoint(f"{config['train']['step-2']['classifier'][0]}", device=device)
    with torch.no_grad():
        net.eval() 
        preds = net(interpolate_list).tolist()

    # save interpolations
    path = f"{os.environ['FILESDIR']}/data/vae/sample_{neg_class}vs{pos_class}_{name}"
    #save_image(interpolate_list, path + ".png", nrow=10)
    torch.save(interpolate_list, f"{path}.pt")
    print(f" > saved interpolation to {path}")

    # save to wandb
    wandb.log({"image_vae": wandb.Image(interpolate_list)})

    wandb.define_metric("prediction")
    for i, pred in enumerate(preds):
        wandb.log({"prediction": pred}, step=i)

    wandb.finish()
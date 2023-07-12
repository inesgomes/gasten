import torch;
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

DEVICE= 'cuda:1'

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
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(DEVICE) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(DEVICE)
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
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(DEVICE) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder

def interpolate(autoencoder, x_1, class_1, x_2, class_2, n=12):
    # create interpolations
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    # save interpolations
    # get most recent file
    path = f"{os.environ['FILESDIR']}/data/vae/"
    files = os.listdir(path)
    no = 1
    if len(files) > 0:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
        no = int(files[-1].split("_")[-1].split(".")[0]) + 1
    path = f"{path}/sample_{class_1}vs{class_2}_{no}"
    save_image(interpolate_list, path + ".png", nrow=10)
    torch.save(interpolate_list, f"{path}.pt")
    print(f" > saved interpolation to {path}")

    return interpolate_list


if __name__ == "__main__":
    load_dotenv()

    # things that could be arguments
    class_1 = 1
    class_2 = 7
    n_interpolations = 10
    classifier = 'cnn-2-1.88299'
    TRAIN = False

    # setup
    latent_dims = 2
    batch_size = 128

    # LOAD data
    print(" > loading data ...")
    # TODO use the same data as the classifier
    # dataset, _, _ = load_dataset(dataset_name, data_dir, pos_class, neg_class, False)

    mnist_dataset = torchvision.datasets.MNIST(f'{os.environ["FILESDIR"]}/data',
                                                    transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                    (0.5,), (0.5,))]),
                                            download=True)

    # Filter the dataset to include only the selected classes
    filtered_indices = torch.where((mnist_dataset.targets == class_1) | (mnist_dataset.targets == class_2))[0]
    filtered_dataset = torch.utils.data.Subset(mnist_dataset, filtered_indices)

    # Create a data loader for the filtered dataset
    data_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

    vae_path = os.environ['FILESDIR'] + f"/models/vae/vae_{class_1}vs{class_2}.pt"
    if os.path.exists(vae_path) & (TRAIN == False):
        vae = torch.load(vae_path)
    else:
        print(" > training VAE ...")
        vae = VariationalAutoencoder(latent_dims).to(DEVICE) # GPU
        vae = train(vae, data_loader)
        torch.save(vae, vae_path)

    print(" > creating interpolations ...")
    x, y = next(data_loader.__iter__()) # hack to grab a batch
    rnd_index = random.randint(0, int(batch_size/10))
    x_1 = x[y == class_1][rnd_index].to(DEVICE) # find a 1
    x_2 = x[y == class_2][rnd_index].to(DEVICE) # find a 7

    interpolate_list = interpolate(vae, x_1, class_1, x_2, class_2, n=n_interpolations)

    # get classifier and visualize probabilities
    print(" > calculating probabilities ...")
    net, _, _, _ = construct_classifier_from_checkpoint(f"{os.environ['FILESDIR']}/models/{classifier}", device=DEVICE)
    net.eval() 
    preds = net(interpolate_list)
    print(preds)

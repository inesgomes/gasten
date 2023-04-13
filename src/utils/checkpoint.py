import os
import json
import torch
import torchvision.utils as vutils
import torch.optim as optim
from src.classifier import construct_classifier
from src.gan import construct_gan
from src.gan.architectures.dcgan import Generator, Discriminator
import matplotlib.pyplot as plt


def checkpoint(model, model_name, model_params, train_stats, args, output_dir=None, optimizer=None):
    output_dir = os.path.curdir if output_dir is None else output_dir
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    save_dict = {
        'name': model_name,
        'state': model.state_dict(),
        'stats': train_stats,
        'params': model_params,
        'args': args
    }

    json.dump({'train_stats': train_stats, 'model_params': model_params, 'args': vars(args)},
              open(os.path.join(output_dir, 'stats.json'), 'w'), indent=2)

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()

    torch.save(save_dict, os.path.join(output_dir, 'classifier.pth'))

    return output_dir


def load_checkpoint(path, model, device=None, optimizer=None):
    cp = torch.load(path, map_location=device)

    model.load_state_dict(cp['state'])
    model.eval()  # just to be safe

    if optimizer is not None:
        optimizer.load_state_dict(cp['optimizer'])


def construct_classifier_from_checkpoint(path, device=None, optimizer=False):
    cp = torch.load(os.path.join(path, 'classifier.pth'), map_location=device)

    print(" > Loading model from {} ...".format(path))

    model_params = cp['params']
    print('\t. Model', cp['name'])
    print('\t. Params: ', model_params)

    n_classes = model_params['n_classes'] if 'n_classes' in model_params else 2

    model = construct_classifier(model_params, device=device)
    model.load_state_dict(cp['state'])
    model.eval()

    if optimizer is True:
        opt = optim.Adam(model.parameters())
        opt.load_state_dict(cp['optimizer'].state_dict())
        return model, model_params, cp['stats'], cp['args'], opt
    else:
        return model, model_params, cp['stats'], cp['args']


def construct_gan_from_checkpoint(path, device=None):
    print("Loading GAN from {} ...".format(path))
    with open(os.path.join(path, 'config.json'), 'r') as config_file:
        config = json.load(config_file)

    model_params = config['model']
    optim_params = config['optimizer']

    gen_cp = torch.load(os.path.join(
        path, 'generator.pth'), map_location=device)
    dis_cp = torch.load(os.path.join(
        path, 'discriminator.pth'), map_location=device)

    G, D = construct_gan(
        model_params, tuple(config['model']['image-size']), device=device)

    g_optim = optim.Adam(G.parameters(), lr=optim_params["lr"], betas=(
        optim_params["beta1"], optim_params["beta2"]))
    d_optim = optim.Adam(D.parameters(), lr=optim_params["lr"], betas=(
        optim_params["beta1"], optim_params["beta2"]))

    G.load_state_dict(gen_cp['state'])
    D.load_state_dict(dis_cp['state'])
    g_optim.load_state_dict(gen_cp['optimizer'])
    d_optim.load_state_dict(dis_cp['optimizer'])

    G.eval()
    D.eval()

    return G, D, g_optim, d_optim


def get_gan_path_at_epoch(output_dir, epoch=None):
    path = output_dir
    if epoch is not None:
        path = os.path.join(path, '{:02d}'.format(epoch))
    return path


def load_gan_train_state(gan_path):
    path = os.path.join(gan_path, 'train_state.json')

    with open(path) as in_f:
        train_state = json.load(in_f)

    return train_state


def checkpoint_gan(G, D, g_opt, d_opt, state, stats, config, output_dir=None, epoch=None):
    rootdir = os.path.curdir if output_dir is None else output_dir

    path = get_gan_path_at_epoch(rootdir, epoch=epoch)

    os.makedirs(path, exist_ok=True)

    torch.save({
        'state': G.state_dict(),
        'optimizer': g_opt.state_dict()
    }, os.path.join(path, 'generator.pth'))

    torch.save({
        'state': D.state_dict(),
        'optimizer': d_opt.state_dict()
    }, os.path.join(path, 'discriminator.pth'))

    json.dump(state, open(os.path.join(
        rootdir, 'train_state.json'), 'w'), indent=2)
    json.dump(stats, open(os.path.join(rootdir, 'stats.json'), 'w'), indent=2)
    json.dump(config, open(os.path.join(path, 'config.json'), 'w'), indent=2)

    print('> Saved checkpoint checkpoint to {}'.format(path))

    return path


def checkpoint_image(image, epoch, output_dir=None):
    dir = os.path.curdir if output_dir is None else output_dir

    dir = os.path.join(dir, 'gen_images')
    os.makedirs(dir, exist_ok=True)

    path = os.path.join(dir, '{:02d}.png'.format(epoch))

    vutils.save_image(image, path)

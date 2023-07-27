"""_summary_ This script is used to run the XAI methods on the saved data.
Raises:
    argparse.ArgumentTypeError: _description_
    argparse.ArgumentTypeError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""
import os
import math
import argparse
import random
from itertools import zip_longest
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from captum.attr import Saliency, IntegratedGradients, DeepLift, NoiseTunnel, GradientShap, Occlusion, LayerGradCam, LayerAttribution, GuidedGradCam
from dotenv import load_dotenv
from src.utils.config import read_config
from src.datasets import load_dataset
from src.datasets.datasets import get_mnist
from src.utils.checkpoint import construct_classifier_from_checkpoint
import pandas as pd
import plotly.express as px
from scipy import stats


def type_data_values(value):
    if value not in ['gasten', 'vae', 'test']:
        raise argparse.ArgumentTypeError(f"Invalid type of data: {value}")
    return value


def sample_no_positive(value):
    if int(value) < 0:
        raise argparse.ArgumentTypeError(f"Invalid sample number: {value}")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        help="Config file", default='experiments/mnist_7v1_1iter.yml')
    parser.add_argument("--type", dest="type_data",
                        help="Type of data [gasten, vae, test]", type=type_data_values)
    parser.add_argument("--no", dest="sample_no",
                        help="Sample number (only for VAE or GASTeN)", type=sample_no_positive)
    parser.add_argument("--random", dest="random",
                        help="If the test set is random or only the selected digits", action="store_true")
    return parser.parse_args()


def get_test_mnist_data(dataset_name, data_dir, batch_size, pos_class=None, neg_class=None):
    # this is the original dataset -> may try to use this initially
    # using test data
    if (pos_class is None) or (neg_class is None):
        dataset = get_mnist(data_dir, train=False)
    else:
        dataset, _, _ = load_dataset(
            dataset_name, data_dir, pos_class, neg_class, False)
    # use data loader to find a batch of images
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    # get first batch of images
    images, _ = next(iter(data_loader))
    return images


def get_saved_data(type_of_data, sample_no, batch_size, pos_class, neg_class):
    # get the images from the saved data
    dataset = torch.load(
        f"{os.environ['FILESDIR']}/data/{type_of_data}/sample_{neg_class}vs{pos_class}_{sample_no}.pt")
    # use data loader to find a batch of images
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    # get first batch of images
    return next(iter(data_loader))


def viz_ecdf(attr, ax, ax_diff):
    # https://plotly.com/python/ecdf-plots/
    # https://matplotlib.org/devdocs/gallery/statistics/histogram_cumulative.html#sphx-glr-gallery-statistics-histogram-cumulative-py

    # transform tensor in dataframe
    attr_flat = np.round(attr.flatten(), decimals=4)
    # print(f" zeros: {len(attr_df_flat[attr_df_flat==0])/len(attr_df_flat):.2%}")

    # create dataframe with two columsn, one with the positive and other with the negative values
    pos = attr_flat[attr_flat > 0]
    neg = np.abs(attr_flat[attr_flat < 0])
    df = pd.DataFrame(
        list(zip_longest(pos, neg, fillvalue=None)), columns=['pos', 'neg'])

    n1, bins, _ = ax.hist(df['pos'], 80, density=True, histtype="step",
            cumulative=-1, label="pos", color="green")
    
    n2, _, _ = ax.hist(df['neg'], bins, density=True, histtype="step",
            cumulative=-1, label="neg", color="red")

    area = abs(n1-n2).sum()
    ax_diff.hist(n1-n2, bins, histtype="stepfilled")
    ax_diff.set_title(f"Area = {area:.2f}")

    return area

    # correct ecdf
    # wandb.log({"ecdf_plotly": px.ecdf(df, x=['pos', 'neg'], ecdfmode="reversed")})


def transform_original_image(image):
    return np.transpose((image.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))


def calc_original(net, input, reference_input, original_image, fig_tuple):
    """
    original image
    """
    pred = net(input)
    viz.visualize_image_attr(None, original_image, method="original_image",
                             title=f"{pred.item():.3f}", plt_fig_axis=fig_tuple)


def calc_saliency(net, input, reference_input, original_image, fig_tuple):
    """_summary_
    Computes gradients with respect to class `ind` and transposes them for visualization purposes.
    Args:
        net (_type_): _description_
        input (_type_): _description_
        labels (_type_): _description_
        ind (_type_): _description_
    """
    # calculate
    saliency = Saliency(net)
    grads = saliency.attribute(input)
    grads_np = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # all saliences
    viz.visualize_image_attr(grads_np, original_image, method="blended_heat_map",
                             sign="absolute_value", show_colorbar=True, plt_fig_axis=fig_tuple)

    return grads_np


def calc_integratedgrads(net, input, reference_input, original_image, fig_tuple):
    """_summary_ 
    TODO: SOLVE, NOT WORKING
    Feature attribution attributes a particular output to features of the input. 
    It uses a specific input to generate a map of the relative importance of each input feature to a particular output feature.
    Integrated Gradients assigns an importance score to each input feature by approximating the integral of the gradients of the model’s 
    output with respect to the inputs.
    Args:
        net (_type_): _description_
        input (_type_): _description_
        original_image (_type_): _description_
        folder_path (_type_): _description_
    """
    ig = IntegratedGradients(net)
    attr_ig, delta = ig.attribute(
        input, reference_input, return_convergence_delta=True, n_steps=300)
    attr_ig = np.transpose(attr_ig.squeeze(
        0).cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    viz.visualize_image_attr(attr_ig, original_image, method="masked_image",
                             sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)
    return attr_ig


def calc_integratedgrads_noise(net, input, reference_input, original_image, fig_tuple):
    ig = IntegratedGradients(net)
    nt = NoiseTunnel(ig)
    attr_ig_nt = nt.attribute(input, baselines=reference_input,
                              nt_type='smoothgrad_sq', nt_samples=300, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    viz.visualize_image_attr(attr_ig_nt, original_image, method="masked_image",
                             sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)
    return attr_ig_nt


def calc_gradientshap(net, input, reference_input, original_image, fig_tuple):
    with torch.no_grad():
        algorithm = GradientShap(net)
        feature_imp_img = algorithm.attribute(input, baselines=reference_input)
    attr = feature_imp_img.squeeze(0).cpu().detach().numpy().reshape(28, 28, 1)

    viz.visualize_image_attr(attr, original_image, method="blended_heat_map",
                             sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)
    return attr


def calc_deeplift(net, input, reference_input, original_image, fig_tuple):
    with torch.no_grad():
        dl = DeepLift(net)
        attr_dl = dl.attribute(input, baselines=reference_input)
    attr_dl = np.transpose(attr_dl.squeeze(
        0).cpu().detach().numpy(), (1, 2, 0))

    # visualize images
    viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",
                             sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)

    return attr_dl

def calc_gradcam_1(net, input, reference_input, original_image, fig_tuple):
    return calc_gradcam(net, input, 1, original_image, fig_tuple)

def calc_gradcam_3(net, input, reference_input, original_image, fig_tuple):
    return calc_gradcam(net, input, 3, original_image, fig_tuple)

def calc_gradcam_5(net, input, reference_input, original_image, fig_tuple):
    return calc_gradcam(net, input, 5, original_image, fig_tuple)

def calc_gradcam_7(net, input, reference_input, original_image, fig_tuple):
    return calc_gradcam(net, input, 7, original_image, fig_tuple)


def calc_gradcam(net, input, layer_idx, original_image, fig_tuple):
    """
    Layer Attribution allows you to attribute the activity of hidden layers within your model to features of your input.
    GradCAM computes the gradients of the target output with respect to the given layer, averages for each output channel, 
    and multiplies the average gradient for each channel by the layer activations.
    """
    layers = list(net.modules())[2:]
    with torch.no_grad():
        layer_gradcam = GuidedGradCam(net, layers[layer_idx])
        attr_lgc = layer_gradcam.attribute(input)

    attr_lgc = np.transpose(attr_lgc.squeeze(
        0).cpu().detach().numpy(), (1, 2, 0))
    # attributions_lgc[0].cpu().permute(1, 2, 0).detach().numpy()

    viz.visualize_image_attr(attr_lgc, original_image, method="blended_heat_map",
                             sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)
    
    return attr_lgc


def calc_occlusion(net, input, reference_input, original_image, fig_tuple):
    """
    It involves replacing sections of the input image, and examining the effect on the output signal
    """
    occlusion = Occlusion(net)
    attributions_occ = occlusion.attribute(input,
                                           sliding_window_shapes=(1, 2, 2),
                                           baselines=reference_input)
    attr = np.transpose(attributions_occ.squeeze(
        0).cpu().detach().numpy(), (1, 2, 0))
    
    viz.visualize_image_attr(attr, original_image, method="blended_heat_map",
                                sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)
    
    return attr


def get_x_y(index, max_y):
    return index // max_y, index % max_y


if __name__ == "__main__":
    ###
    # Setup
    ###

    # load environment variables, arguments and configs
    load_dotenv()
    args = parse_args()
    config = read_config(args.config_path)

    if (args.type_data != "test") & (args.sample_no is None):
        raise ValueError("You must provide a sample number")

    device = torch.device(config["device"])
    pos_class = config["dataset"]["binary"]["pos"]
    neg_class = config["dataset"]["binary"]["neg"]
    config_run = {
        'batch_size': 10,
        'save_internally': False,
        'classifier': config['train']['step-2']['classifier'][0].split("/")[-1],
    }

    # start experiment
    name = args.sample_no if args.type_data != "test" else random.randint(
        1, 10000)
    type_name = "test_random" if (args.random) & (
        args.type_data == "test") else args.type_data
    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='xai',
               name=f"{config_run['classifier']}-{type_name}_{name}_tst",
               config=config_run)

    # get data
    if args.type_data == "test":
        if args.random:
            images = get_test_mnist_data(
                config["dataset"]["name"], config["data-dir"], config_run['batch_size'])
        else:
            images = get_test_mnist_data(
                config["dataset"]["name"], config["data-dir"], config_run['batch_size'], pos_class, neg_class)
    else:
        images = get_saved_data(
            args.type_data, args.sample_no, config_run['batch_size'], pos_class, neg_class)

    # get classifier
    net, _, _, _ = construct_classifier_from_checkpoint(
        config['train']['step-2']['classifier'][0], device=device)
    net.eval()

    # prepare the wandb plots
    max_y = 5
    max_x, _ = get_x_y(images.shape[0], max_y)

    # available methods
    dict = {
        'original': calc_original,
        'saliency': calc_saliency,
        #'integratedgrads': calc_integratedgrads,
        'gradientshap': calc_gradientshap,
        'deeplift': calc_deeplift,
        'gradcam_1': calc_gradcam_1,
        'occlusion': calc_occlusion,
    }

    # prepare image data
    inputs = []
    original_images = []
    for ind, image in enumerate(images):
        inputs.append(image.unsqueeze(0).to(device))
        original_images.append(transform_original_image(image))
    # reference image
    background = 0 if args.type_data == "vae" else -1
    reference_input = torch.full(
        images[0].shape, background).unsqueeze(0).to(device)
    # predictions
    preds = net(images).detach().cpu().numpy()

    # prepare correlations table
    test_table = wandb.Table(columns=['method', 'correlation', 'p-value', 'n'])

    # run methods
    not_ecdf_methods = ['original', 'saliency', 'gradcam_1']
    for name, method in dict.items():
        print("Running method: ", name)
        # prepare plots
        fig, axes = plt.subplots(max_x, max_y, figsize=(12, 2*max_x))
        if name not in not_ecdf_methods:
             fig_ecdf, axes_ecdf = plt.subplots(
                max_x, max_y, figsize=(12, 2*max_x), sharex=True, sharey=True)
             fig_ecdf_diff, axes_ecdf_diff = plt.subplots(
                max_x, max_y, figsize=(12, 2*max_x), sharex=True, sharey=True)

        areas = []
        for ind in range(images.shape[0]):
            # prepare indexes and paths
            x, y = get_x_y(ind, max_y)
            # calculate attributions
            attr = method(net, inputs[ind], reference_input,
                          original_images[ind], (fig, axes[x][y]))
            # calculate ecdf
            if name not in not_ecdf_methods:
               area = viz_ecdf(attr, axes_ecdf[x][y], axes_ecdf_diff[x][y])
               areas.append(area)
            del attr

        # save images and ecdf to wandb
        wandb.log({"image_xai": wandb.Image(fig, caption=name)})
        if name not in not_ecdf_methods:
            # print ECDFs
            fig_ecdf.supxlabel(f"{name} attributions (absolute value)")
            fig_ecdf.supylabel("Likelihood of attribution")
            fig_ecdf.tight_layout()
            wandb.log({"ecdf": wandb.Image(fig_ecdf, caption=f"Complementary cumulative distributions for positive and negative {name} attributions")})
            wandb.log({"ecdf_diff": wandb.Image(fig_ecdf_diff, caption=f"Area under positive and negative {name} attributions")})            
            del fig_ecdf, axes_ecdf, fig_ecdf_diff, axes_ecdf_diff

            # check correlation between areas and predictions
            corr, pvalue = stats.spearmanr(np.array(areas), preds)
            test_table.add_data(name, corr, pvalue, len(areas))

        del fig, axes

    wandb.log({"correlations": test_table})
    # close wandb
    wandb.finish()

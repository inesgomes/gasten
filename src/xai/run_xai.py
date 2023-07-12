
import os
import math
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from captum.attr import Saliency, IntegratedGradients, DeepLift, NoiseTunnel, GradientShap, Occlusion, LayerGradCam, LayerAttribution
from dotenv import load_dotenv
from src.utils.config import read_config
from src.datasets import load_dataset
from src.utils.checkpoint import construct_classifier_from_checkpoint
import argparse
import random


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
    parser.add_argument("--config", dest="config_path",  help="Config file", default='experiments/mnist_7v1_1iter.yml')
    parser.add_argument("--type", dest="type_data", help="Type of data [gasten, vae, test]", type=type_data_values)
    parser.add_argument("--no", dest="sample_no", help="Sample number (only for VAE or GASTeN)", type=sample_no_positive)
    return parser.parse_args()


def get_test_mnist_data(dataset_name, data_dir, batch_size, pos_class, neg_class):
    # this is the original dataset -> may try to use this initially
    # using test data 
    dataset, _, _ = load_dataset(dataset_name, data_dir, pos_class, neg_class, False)
    # use data loader to find a batch of images
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # get first batch of images
    images, _ = next(iter(data_loader))
    return images

def get_saved_data(type_of_data, sample_no, batch_size, pos_class, neg_class):
    # get the images from the saved data
    dataset = torch.load(f"{os.environ['FILESDIR']}/data/{type_of_data}/sample_{neg_class}vs{pos_class}_{sample_no}.pt")
    # use data loader to find a batch of images
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # get first batch of images
    return next(iter(data_loader))


def calc_original(image, folder_path):
    """
    original image
    """
    # calculate
    original_image = np.transpose((image.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    # visualize
    viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f"{folder_path}/original.png")
    plt.close()

    return original_image
    

def calc_saliency(net, input, original_image, folder_path, fig_tuple):
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

    # visualize
    viz.visualize_image_attr(grads_np, original_image, method="blended_heat_map", sign="absolute_value", show_colorbar=True, title="Overlayed Gradient Magnitudes")
    plt.savefig(f"{folder_path}/saliency.png")
    plt.close()

    # all saliences
    viz.visualize_image_attr(grads_np, original_image, method="blended_heat_map", sign="absolute_value", show_colorbar=True, plt_fig_axis=fig_tuple)


def calc_integratedgrads(net, input, original_image, folder_path, fig_tuple):
    """_summary_
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
    attr_ig, delta = ig.attribute(input, baselines=input * 0, return_convergence_delta=True, n_steps=200)
    attr_ig = np.transpose(attr_ig.squeeze(0).cpu().detach().numpy(), (1,2,0))
    print('Approximation delta: ', abs(delta))
    
    # visualize
    viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")
    plt.savefig(f"{folder_path}/integrated_grads.png")
    plt.close()

    viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)


def calc_integratedgrads_noise(net, input, original_image, folder_path):
    ig = IntegratedGradients(net)
    nt = NoiseTunnel(ig)
    attr_ig_nt = nt.attribute(input, baselines=input * 0, nt_type='smoothgrad_sq', nt_samples=300, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # visualize
    viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="all", show_colorbar=True, outlier_perc=10,
                             title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
    plt.savefig(f"{folder_path}/integrated_grads_noise.png")
    plt.close()
    

def calc_gradientshap(net, input, original_image, folder_path, fig_tuple):
    algorithm = GradientShap(net)
    feature_imp_img = algorithm.attribute(input, baselines=torch.zeros_like(input))
    attr = feature_imp_img.squeeze(0).cpu().detach().numpy().reshape(28,28,1)

    viz.visualize_image_attr(attr, original_image, method="blended_heat_map", sign="all", show_colorbar=True, title="Gradient SHAP")
    plt.savefig(f"{folder_path}/gradientSHAP.png")
    plt.close()

    viz.visualize_image_attr(attr, original_image, method="blended_heat_map", sign="all", show_colorbar=True, plt_fig_axis=fig_tuple)


def calc_deeplift(net, input, original_image, folder_path, fig_tuple):
    dl = DeepLift(net)
    attr_dl = dl.attribute(input, baselines=input * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
                          title="Overlayed DeepLift")
    plt.savefig(f"{folder_path}/deepLift.png")
    plt.close()

    viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, plt_fig_axis=fig_tuple)


def calc_occlusion(net, input, original_image, folder_path):
    """
    It involves replacing sections of the input image, and examining the effect on the output signal
    NOT WORKING
    """
    occlusion = Occlusion(net)
    attributions_occ = occlusion.attribute(input,
                                       strides=(3, 8, 8),
                                       sliding_window_shapes=(1, 15, 15),
                                       baselines=0)
    attr = np.transpose(attributions_occ.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    
    viz.visualize_image_attr_multiple(attr,
                                      original_image,
                                      ["original_image", "heat_map", "heat_map", "masked_image"],
                                      ["all", "positive", "negative", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                     )
    plt.savefig(f"{folder_path}/occlusion.png")
    plt.close()


def calc_gradcam(net, layer_idx, input, original_image, folder_path):
    """
    Layer Attribution allows you to attribute the activity of hidden layers within your model to features of your input.
    GradCAM computes the gradients of the target output with respect to the given layer, averages for each output channel, 
    and multiplies the average gradient for each channel by the layer activations.
    """
    layers = list(net.modules())[2:]
    #print(len(layers))
    layer_gradcam = LayerGradCam(net, layers[layer_idx])
    attributions_lgc = layer_gradcam.attribute(input)
    viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(), sign="all",  title="Layer {layer_idx}")
    plt.savefig(f"{folder_path}/layer{layer_idx}_gradcam.png")
    plt.close()

    upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input.shape[2:])

    viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                      original_image,
                                      ["original_image","blended_heat_map","masked_image"],
                                      ["all","positive","positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      )
    plt.savefig(f"{folder_path}/layer{layer_idx}_gradcam_mult.png")
    plt.close()

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
    config_run={
        'batch_size': 10,
        'classifier': config['train']['step-2']['classifier'][0].split("/")[-1],
    }

    # start experiment
    name = args.sample_no if args.type_data != "test" else random.randint(1, 10000)
    wandb.init(project=config['project'],
               dir=os.environ['FILESDIR'],
               group=config['name'],
               entity=os.environ['ENTITY'],
               job_type='xai',
               name=f"{args.type_data}-{name}",
               config=config_run)
    
    # get data
    if args.type_data == "test":
        images = get_test_mnist_data(config["dataset"]["name"], config["data-dir"], config_run['batch_size'], pos_class, neg_class)
    else:
        images = get_saved_data(args.type_data, args.sample_no, config_run['batch_size'], pos_class, neg_class)
    
    # get classifier 
    net, _, _, _ = construct_classifier_from_checkpoint(config['train']['step-2']['classifier'][0], device=device)
    net.eval() 

    # prepare the wandb plots
    max_x, _ = get_x_y(images.shape[0], config_run['batch_size'])
    fig_ori, axes_ori = plt.subplots(max_x, config_run['batch_size'], figsize=(16,3*max_x))
    fig_sal, axes_sal = plt.subplots(max_x, config_run['batch_size'], figsize=(16,3*max_x))
    fig_ig, axes_ig = plt.subplots(max_x, config_run['batch_size'], figsize=(16,3*max_x))
    fig_shap, axes_shap = plt.subplots(max_x, config_run['batch_size'], figsize=(16,3*max_x))
    fig_dl, axes_dl = plt.subplots(max_x, config_run['batch_size'], figsize=(16,3*max_x))

    for ind in range(images.shape[0]):
        # prepare data to make predictions
        image = images[ind].to(device)
        input = image.unsqueeze(0)
        #input.requires_grad = True

        # predict
        pred = net(input)
        label = pos_class if pred >= 0.5 else neg_class
        
        # prepare indexes and paths
        x, y = get_x_y(ind, config_run['batch_size'])
        folder_path = f"{config['data-dir']}/xai/{config_run['classifier']}/{args.type_data}{name}/{ind}_pred_{label}_{math.floor(pred.item()*100)}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(f" > Saving to {folder_path}")

        # calculate interpretable representation
        original_image = calc_original(image, folder_path)
        viz.visualize_image_attr(None, original_image, method="original_image", title=f"{pred.item():.3f}", plt_fig_axis=(fig_ori, axes_ori[x][y] if x > 1 else axes_ori[y]))
        calc_saliency(net, input, original_image, folder_path, (fig_sal, axes_sal[x][y] if x > 1 else axes_sal[y]))
        net.zero_grad()
        calc_integratedgrads(net, input, original_image, folder_path, (fig_ig, axes_ig[x][y] if x > 1 else axes_ig[y]))
        calc_integratedgrads_noise(net, input, original_image, folder_path)
        calc_gradientshap(net, input, original_image, folder_path, (fig_shap, axes_shap[x][y] if x > 1 else axes_shap[y]))
        calc_deeplift(net, input, original_image, folder_path, (fig_dl, axes_dl[x][y] if x > 1 else axes_dl[y]))
        calc_gradcam(net, 0, input, original_image, folder_path)
        calc_gradcam(net, 3, input, original_image, folder_path)
        calc_gradcam(net, 7, input, original_image, folder_path)

    # save images to wandb
    wandb.log({"image_xai": wandb.Image(fig_ori, caption="Original Images")})
    wandb.log({"image_xai": wandb.Image(fig_sal, caption="Saliency Method")})
    wandb.log({"image_xai": wandb.Image(fig_ig, caption="Integrated Gradients Method")})
    wandb.log({"image_xai": wandb.Image(fig_shap, caption="Gradient SHAP Method")})
    wandb.log({"image_xai": wandb.Image(fig_dl, caption="DeepLift Method")})

    # close wandb
    wandb.finish()


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
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_CMAP = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

def get_test_mnist_data(dataset_name, data_dir, ind, pos_class, neg_class):
    # this is the original dataset -> may try to use this initially
    # using test data 
    dataset, _, _ = load_dataset(dataset_name, data_dir, pos_class, neg_class, False)
    images = dataset.data.to(device)
    return images[ind]


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
    plt.savefig(f"{folder_path}/original.svg")
    plt.close()

    return original_image
    

def calc_saliency(net, input, original_image, folder_path):
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
    plt.savefig(f"{folder_path}/saliency.svg")
    plt.close()


def calc_integratedgrads(net, input, original_image, folder_path):
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
    plt.savefig(f"{folder_path}/integrated_grads.svg")
    plt.close()


def calc_integratedgrads_noise(net, input, original_image, folder_path):
    ig = IntegratedGradients(net)
    nt = NoiseTunnel(ig)
    attr_ig_nt = nt.attribute(input, baselines=input * 0, nt_type='smoothgrad_sq', nt_samples=300, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # visualize
    viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="all", show_colorbar=True, outlier_perc=10,
                             title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
    plt.savefig(f"{folder_path}/integrated_grads_noise.svg")
    plt.close()
    

def calc_gradientshap(net, input, original_image, folder_path):
    algorithm = GradientShap(net)
    feature_imp_img = algorithm.attribute(input, baselines=torch.zeros_like(input))
    attr = feature_imp_img.squeeze(0).cpu().detach().numpy().reshape(28,28,1)

    viz.visualize_image_attr(attr, original_image, method="blended_heat_map", show_colorbar=True, title="Gradient SHAP")
    plt.savefig(f"{folder_path}/gradientSHAP.svg")
    plt.close()


def calc_deeplift(net, input, original_image, folder_path):
    dl = DeepLift(net)
    attr_dl = dl.attribute(input, baselines=input * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
                          title="Overlayed DeepLift")
    plt.savefig(f"{folder_path}/deepLift.svg")
    plt.close()


def calc_occlusion(net, input, original_image, folder_path):
    """
    It involves replacing sections of the input image, and examining the effect on the output signal
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
    plt.savefig(f"{folder_path}/occlusion.svg")
    plt.close()


def calc_gradcam(net, layer_idx, input, original_image, folder_path):
    """
    Layer Attribution allows you to attribute the activity of hidden layers within your model to features of your input.
    GradCAM computes the gradients of the target output with respect to the given layer, averages for each output channel, 
    and multiplies the average gradient for each channel by the layer activations.
    """
    layers = list(net.modules())[2:]
    print(len(layers))

    layer_gradcam = LayerGradCam(net, layers[layer_idx])
    attributions_lgc = layer_gradcam.attribute(input)
    viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(), sign="all",  title="Layer {layer_idx}")
    plt.savefig(f"{folder_path}/layer{layer_idx}_gradcam.svg")
    plt.close()

    upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input.shape[2:])

    viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                      original_image,
                                      ["original_image","blended_heat_map","masked_image"],
                                      ["all","positive","positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      )
    plt.savefig(f"{folder_path}/layer{layer_idx}_gradcam_mult.svg")
    plt.close()


if __name__ == "__main__":
    # load environment variables
    load_dotenv()
    # read configs
    config = read_config('experiments/mnist_7v1.yml')

    # WANDB
    #api = wandb.Api()
    #file = api.run(f"gomes-inesisabel/gasten_20230413/2wjbt22x").file("media/images/eval/samples_79_ec085773881b0d6e11df.png")
    #print(file.type)

    ###
    # Setup
    ###
    pos_class = config["dataset"]["binary"]["pos"]
    neg_class = config["dataset"]["binary"]["neg"]
    model_name = config['train']['step-2']['classifier'][2].split("/")[-1]
    device = torch.device(config["device"])

    # get classifier 
    net, _, _, _ = construct_classifier_from_checkpoint(config['train']['step-2']['classifier'][2], device=device)
    net.eval() 

    # TEST SET DATA
    ind = 0
    image = get_test_mnist_data(config["dataset"]["name"], config["data-dir"], ind, pos_class, neg_class)

    input = image.unsqueeze(0)
    input.requires_grad = True

    pred = net(input)
    label = pos_class if pred >= 0.5 else neg_class
    folder_path = f"{config['data-dir']}/xai/{model_name}/mnist_{label}_{math.floor(pred.item()*100)}"
    print(f" > Saving to {folder_path}")
    
    # calculate interpretable representation
    original_image = calc_original(images[ind], folder_path)
    calc_saliency(net, input, original_image, folder_path)
    net.zero_grad()
    calc_integratedgrads(net, input, original_image, folder_path)
    calc_integratedgrads_noise(net, input, original_image, folder_path)
    calc_gradientshap(net, input, original_image, folder_path)
    calc_deeplift(net, input, original_image, folder_path)
    # not working
    #calc_occlusion(net, input, original_image, folder_path)
    calc_gradcam(net, 0, input, original_image, folder_path)
    calc_gradcam(net, 3, input, original_image, folder_path)
    calc_gradcam(net, 7, input, original_image, folder_path)

    # TODO the remaining visualizations  
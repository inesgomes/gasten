from src.gan.architectures.dcgan import Generator as DC_G, Discriminator as DC_D
from src.gan.architectures.resnet import Generator as RN_G, Discriminator as RN_D
from src.gan.loss import NS_DiscriminatorLoss, NS_GeneratorLoss, W_GeneratorLoss, WGP_DiscriminatorLoss


def construct_gan(config, img_size, device):
    use_batch_norm = config["loss"]["name"] != 'wgan-gp'
    is_critic = config["loss"]["name"] == 'wgan-gp'

    arch_config = config["architecture"]

    if arch_config["name"] == "dcgan":
        G = DC_G(img_size, z_dim=config['z_dim'],
                 filter_dim=arch_config['g_filter_dim'],
                 n_blocks=arch_config['g_num_blocks']).to(device)

        D = DC_D(img_size, filter_dim=arch_config['d_filter_dim'],
                 n_blocks=arch_config['d_num_blocks'],
                 use_batch_norm=use_batch_norm, is_critic=is_critic).to(device)
    elif arch_config["name"] == "resnet":
        G = RN_G(img_size, z_dim=config['z_dim'],
                 gf_dim=arch_config['g_filter_dim']).to(device)
        D = RN_D(img_size, df_dim=arch_config['d_filter_dim'],
                 use_batch_norm=use_batch_norm, is_critic=is_critic).to(device)

    return G, D


def construct_loss(config, D):
    if config["name"] == "ns":
        return NS_GeneratorLoss(), NS_DiscriminatorLoss()
    elif config["name"] == "wgan-gp":
        return W_GeneratorLoss(), WGP_DiscriminatorLoss(D, config["args"]["lambda"])

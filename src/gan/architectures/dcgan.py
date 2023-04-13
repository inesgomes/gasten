import torch.nn as nn
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Same') != -1:
        nn.init.normal_(m.conv_t_2d.weight.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def conv_out_size_same(size, stride):
    return int(np.ceil(float(size) / float(stride)))


def compute_padding_same(in_size, out_size, kernel, stride):
    # Hout​= (Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1

    # 2 * padding - out_padding =
    res = (in_size - 1) * stride - out_size + kernel

    out_padding = 0 if (res % 2 == 0) else 1
    padding = (res + out_padding) / 2

    return int(padding), int(out_padding)


class ConvTranspose2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size, kernel, stride, bias=True):
        super(ConvTranspose2dSame, self).__init__()

        in_h, in_w = in_size
        out_h, out_w = out_size

        pad_h, out_pad_h = compute_padding_same(in_h, out_h, kernel, stride)
        pad_w, out_pad_w = compute_padding_same(in_w, out_w, kernel, stride)

        pad = (pad_h, pad_w)
        out_pad = (out_pad_h, out_pad_w)

        self.conv_t_2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel, stride, pad, out_pad, bias=bias)

    def forward(self, x):
        return self.conv_t_2d(x)


class Generator(nn.Module):
    def __init__(self, image_size, z_dim=100, n_blocks=3, filter_dim=64):
        """
        nz: size of latent code
        ngf: dimension of filters in first convolutional layer
        """
        super(Generator, self).__init__()
        self.image_size = image_size
        self.z_dim = z_dim
        self.filter_dim = filter_dim
        self.n_blocks = n_blocks

        n_channels, cur_s_h, cur_s_w = image_size
        conv_blocks_rev = nn.ModuleList()

        for i in range(n_blocks):
            cur_s_h_smaller = conv_out_size_same(cur_s_h, 2)
            cur_s_w_smaller = conv_out_size_same(cur_s_w, 2)

            if i == 0:
                block = nn.Sequential(
                    ConvTranspose2dSame(
                        filter_dim, n_channels,
                        (cur_s_h_smaller, cur_s_w_smaller),
                        (cur_s_h, cur_s_w),
                        5, 2, bias=False),
                    nn.Tanh(),
                )
            else:
                in_channels = filter_dim * 2 ** i
                out_channels = filter_dim * 2 ** (i - 1)

                block = nn.Sequential(
                    ConvTranspose2dSame(
                        in_channels, out_channels,
                        (cur_s_h_smaller, cur_s_w_smaller),
                        (cur_s_h, cur_s_w),
                        5, 2, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )

            cur_s_h, cur_s_w = cur_s_h_smaller, cur_s_w_smaller

            conv_blocks_rev.append(block)

        project_out_dim = filter_dim * \
            (2 ** (n_blocks - 1)) * cur_s_h * cur_s_w

        self.project_out_reshape_dim = (
            filter_dim * (2 ** (n_blocks - 1)), cur_s_h, cur_s_w)

        self.project = nn.Sequential(
            nn.Linear(z_dim, project_out_dim, bias=False),
            nn.BatchNorm1d(project_out_dim),
            nn.ReLU(True),
        )

        self.conv_blocks = nn.Sequential()
        for i in reversed(conv_blocks_rev):
            self.conv_blocks.append(i)

        self.apply(weights_init)

    def forward(self, z):
        """
        z: input (batch_size, z_dim)
        """
        z = self.project(z)
        z = self.conv_blocks(z.view(-1, *self.project_out_reshape_dim))

        return z


class Discriminator(nn.Module):
    def __init__(self, image_size, n_blocks=2, filter_dim=64, use_batch_norm=True, is_critic=False):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.filter_dim = filter_dim
        self.n_blocks = n_blocks

        n_channels, cur_s_h, cur_s_w = image_size

        self.conv_blocks = nn.Sequential()

        for i in range(n_blocks):
            cur_s_h = conv_out_size_same(cur_s_h, 2)
            cur_s_w = conv_out_size_same(cur_s_w, 2)

            out_channels = filter_dim * 2 ** i
            if i == 0:
                in_channels = n_channels
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                in_channels = filter_dim * 2 ** (i - 1)
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False),
                    nn.BatchNorm2d(
                        out_channels) if use_batch_norm else nn.LayerNorm([out_channels, cur_s_h, cur_s_w]),
                    nn.LeakyReLU(0.2, inplace=True)
                )

            self.conv_blocks.append(block)

        self.predict = nn.Sequential(
            # (b, ndf * 2, 7, 7)
            nn.Flatten(),
            nn.Linear(filter_dim * 2 ** (n_blocks - 1) * cur_s_h * cur_s_w, 1),
            # (b, 1)
        )

        if not is_critic:
            self.predict.append(nn.Sigmoid())

        self.apply(weights_init)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        return self.predict(x).squeeze()

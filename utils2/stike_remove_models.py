import functools

import torch
import torch.nn as nn
from torchvision.models.densenet import _DenseLayer

"""
The DenseGenerator architecture is based on the ResnetGenerator provided by Yhu et al,
taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

See also:

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
"""


class DenseGenerator(nn.Module):
    """
    Based on ResnetGenerator, provided by Yhu et al, taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Modification replaces resnet blocks with dense ones.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6):
        """Construct a generator with a dense bottleneck.
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            n_blocks (int)      -- the number of Dense blocks
        """
        assert (n_blocks >= 0)
        super(DenseGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        dense_features = ngf * mult
        dense_features = dense_features + 6 * 32
        for i in range(n_blocks):  # add ResNet blocks
            model += [DenseBlock(num_layers=6, num_input_features=ngf * mult, bn_size=4, growth_rate=32, drop_rate=0,
                                 norm_layer=norm_layer)]
            model += [norm_layer(dense_features), nn.ReLU(inplace=True),
                      nn.Conv2d(dense_features, ngf * mult, kernel_size=1, stride=1, bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Shallow(nn.Module):
    """
    Based on ResnetGenerator, provided by Yhu et al, taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Modification replaces resnet blocks with dense ones.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        """Construct a generator without a dedicated bottleneck.
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(Shallow, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]

        model += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2),
                  nn.ReLU(True)]

        model += [
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class SimpleCNN(nn.Module):
    """
    Small autoencoder with no dedicated bottleneck layer.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU()
                                     )

        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())
        self.decoder.apply(init_weights)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DenseBlock(nn.ModuleDict):
    """
    Based on torchvision.models.densenet, extended with an option to specify the normalisations for norm1 and norm2
    """
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False,
                 norm_layer=nn.BatchNorm2d):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size,
                                drop_rate=drop_rate, memory_efficient=memory_efficient)
            if norm_layer != nn.BatchNorm2d:
                layer.norm1 = norm_layer(num_input_features)
                layer.norm2 = norm_layer(bn_size * growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

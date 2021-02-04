import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict


class DenseLayer(nn.Module):
    def __init__(self, in_c, growth_rate, drop_rate):
        super(DenseLayer, self).__init__()

        layers = OrderedDict([
            ("norm1", nn.BatchNorm2d(in_c)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv1", nn.Conv2d(in_c, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
            ("drop1", nn.Dropout(drop_rate))
        ])

        for name, layer in layers.items():
            self.add_module(name, layer)

    def forward(self, inps):
        if isinstance(inps, Tensor):
            inps = [inps]
        else:
            inps = inps

        out = torch.cat(inps, dim=1)
        for layer in self.children():
            out = layer(out)
        return out

class DenseLayer_BC(nn.Module):
    def __init__(self, in_c, growth_rate, bn_size, drop_rate):
        super(DenseLayer_BC, self).__init__()

        layers = OrderedDict([
            ("norm1", nn.BatchNorm2d(in_c)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv1", nn.Conv2d(in_c, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
            ("drop1", nn.Dropout2d(drop_rate)),

            ("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
            ("relu2", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
            ("drop2", nn.Dropout2d(drop_rate))
        ])

        for name, layer in layers.items():
            self.add_module(name, layer)

    def forward(self, inps):
        if isinstance(inps, Tensor):
            inps = [inps]
        else:
            inps = inps

        out = torch.cat(inps, dim=1)
        for layer in self.children():
            out = layer(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_c, num_layer, bn_size, growth_rate, drop_rate, bn=True):
        super(DenseBlock, self).__init__()

        layers = []
        for n in range(num_layer):
            if bn:
                layer = DenseLayer_BC(
                    in_c + n * growth_rate,
                    growth_rate,
                    bn_size,
                    drop_rate
                )
                layers.append(layer)
            else:
                layer = DenseLayer(
                    in_c + n * growth_rate,
                    growth_rate,
                    drop_rate
                )
                layers.append(layer)

        layers = nn.ModuleList(layers)
        for idx, layer in enumerate(layers):
            self.add_module("DenseLayer{}".format(idx+1), layer)

    def forward(self, inps):
        inps = [inps]

        for layer in self.children():
            new_inp = layer(inps)
            inps.append(new_inp)

        return torch.cat(inps, 1)


class Transition(nn.Module):
    def __init__(self, in_c, theta=0.5):
        super(Transition, self).__init__()

        layers = OrderedDict([
            ("norm", nn.BatchNorm2d(in_c)),
            ("relu", nn.ReLU(True)),
            ("conv", nn.Conv2d(in_c, int(in_c * theta), kernel_size=1, stride=1, bias=False)),
            ("pool", nn.AvgPool2d(kernel_size=2, stride=2))
        ])

        for name, layer in layers.items():
            self.add_module(name, layer)

    def forward(self, inps):
        outs = inps
        for layer in self.children():
            outs = layer(outs)

        return outs


class DenseNet_CIFAR(nn.Module):
    def __init__(self,
                 in_c=3,
                 bn_size=4,
                 growth_rate=12,
                 block_nums=[16, 16, 16],
                 init_feature=16,
                 drop_rate=0.2,
                 cls_nums=10,
                 bn=True):

        super(DenseNet_CIFAR, self).__init__()
        theta = 0.5 if bn else 1.0
        init_feature = 2*growth_rate if bn else init_feature

        self.head = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(in_c, init_feature, kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ("norm1", nn.BatchNorm2d(init_feature)),
                ("relu1", nn.ReLU(True)),
            ])
        )

        feature_num = init_feature


        for n, layer_num in enumerate(block_nums):
            tmp_block = DenseBlock(
                in_c=feature_num,
                num_layer=layer_num,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                bn=bn
            )

            self.add_module("DenseBlock{}".format(n+1), tmp_block)
            feature_num = feature_num + layer_num * growth_rate

            if n == len(block_nums) - 1:
                break
            else:
                tmp_trans = Transition(in_c=feature_num, theta=theta)
                self.add_module("Transition{}".format(n+1), tmp_trans)
                feature_num = int(feature_num * theta)

        self.last = nn.Sequential(
            OrderedDict([
                ('norm4', nn.BatchNorm2d(feature_num)),
                ('relu4', nn.ReLU(True)),
                ('pool4', nn.AdaptiveAvgPool2d((1, 1))),
                ('flttn', nn.Flatten()),
                ('classifier', nn.Linear(feature_num, cls_nums))
            ])
        )

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        return out

def initialize(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)


def densenet_40(init=True):
    model = DenseNet_CIFAR(block_nums=[12, 12, 12],  growth_rate=12, bn=False)
    if init:
        model.apply(initialize)
    return model

def densenet_100_12(init=True):
    model = DenseNet_CIFAR(block_nums=[32, 32, 32],  growth_rate=12, bn=False)
    if init:
        model.apply(initialize)
    return model

def densenet_100_24(init=True):
    model = DenseNet_CIFAR(block_nums=[32, 32, 32],  growth_rate=24, bn=False)
    if init:
        model.apply(initialize)
    return model

def densenet_bc_100(init=True):
    model = DenseNet_CIFAR(block_nums=[16, 16, 16],  growth_rate=12, bn=True)
    if init:
        model.apply(initialize)
    return model

def densenet_bc_250(init=True):
    model = DenseNet_CIFAR(block_nums=[41, 41, 41],  growth_rate=24, bn=True)
    if init:
        model.apply(initialize)
    return model

def densenet_bc_190(init=True):
    model = DenseNet_CIFAR(block_nums=[31, 31, 31],  growth_rate=40, bn=True)
    if init:
        model.apply(initialize)
    return model


if __name__ == "__main__":
    from torchinfo import summary
    
    model_size = ['40'] #, '100_12', '100_24', 'bc_100', 'bc_190', 'bc_250']

    for size in model_size:
        my = eval('densenet_' + size)()
        summary(my)


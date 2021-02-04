import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict


class DenseLayer(nn.Module):
    def __init__(self, in_c, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()

        layers = [
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.Dropout2d(drop_rate),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout2d(drop_rate)
        ]
        self.dense = nn.Sequential(*layers)

    def forward(self, inps):
        if isinstance(inps, Tensor):
            inps = [inps]
        else:
            inps = inps

        out = torch.cat(inps, dim=1)
        out = self.dense(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_c, num_layer, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()

        self.layers = []
        for n in range(num_layer):
            layer = DenseLayer(
                in_c + n * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, inps):
        inps = [inps]

        for layer in self.layers:
            new_inp = layer(inps)
            inps.append(new_inp)

        return torch.cat(inps, 1)


class Transition(nn.Module):
    def __init__(self, in_c, theta=0.5):
        super(Transition, self).__init__()

        layers = [
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_c, int(in_c * theta), kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ]

        self.transit = nn.Sequential(
            *layers
        )

    def forward(self, inps):
        outs = inps

        outs = self.transit(outs)
        return outs


class DenseNet(nn.Module):
    def __init__(
            self,
            bn_size=4,
            growth_rate=32,
            block_nums=[6, 12, 24, 16],
            init_feature=64,
            drop_rate=0.2,
            cls_nums=1000
    ):

        super(DenseNet, self).__init__()
        self.head = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, init_feature, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ("norm1", nn.BatchNorm2d(init_feature)),
                ("relu1", nn.ReLU(True)),
                ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])
        )

        self.blocks = []
        feature_num = init_feature
        for n, layer_num in enumerate(block_nums):
            tmp_block = DenseBlock(
                in_c=feature_num,
                num_layer=layer_num,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.blocks.append(tmp_block)
            feature_num = feature_num + layer_num * growth_rate

            if n == len(block_nums) - 1:
                break
            else:
                tmp_trans = Transition(in_c=feature_num)
                self.blocks.append(tmp_trans)
                feature_num = feature_num // 2


        self.blocks.extend([
            nn.BatchNorm2d(feature_num),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        ])

        self.blocks = nn.ModuleList(self.blocks)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(feature_num, cls_nums)

    def forward(self, x):
        out = self.head(x)

        for layer in self.blocks:
            out = layer(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


def initialize(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)


def densenet_121(init=True):
    model = DenseNet()
    if init:
        model.apply(initialize)
    return model


def densenet_169(init=True):
    model = DenseNet(block_nums=[6, 12, 32, 32])
    if init:
        model.apply(initialize)
    return model


def densenet_201(init=True):
    model = DenseNet(block_nums=[6, 12, 48, 32])
    if init:
        model.apply(initialize)
    return model


def densenet_264(init=True):
    model = DenseNet(block_nums=[6, 12, 64, 48])
    if init:
        model.apply(initialize)
    return model


if __name__ == "__main__":
    import torchvision
    from torchinfo import summary
    
    model_sizes = [121, 169, 201, 264]
    
    for size in model_sizes:
        my = eval('densenet_' + str(size))()
        original = eval('torchvision.models.densenet' + str(size))()
    
        summary(my)
        summary(original)
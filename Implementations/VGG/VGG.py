import torch
import torch.nn as nn


# Define Activation Function
def activation_fun(activ):
    return nn.ModuleDict([
        ['relu', nn.ReLU(True)],
        ['leaky_relu', nn.LeakyReLU(0.02, True)],
        ['selu', nn.SELU(True)],
        ['none', nn.Identity()]
    ])[activ]


# Define Basic block for VGG
def basic_block(in_c, out_c, k, depth, activation='relu', bn=False):
    blocks = []
    for _ in range(depth):
        if _ == 0:
            blocks.append(nn.Conv2d(in_c, out_c, kernel_size=k, padding=k // 2, bias=True))
        else:
            blocks.append(nn.Conv2d(out_c, out_c, kernel_size=k, padding=k // 2, bias=True))

        if bn:
            blocks.append(nn.BatchNorm2d(out_c))
        blocks.append(activation_fun(activation))
    blocks.append(nn.MaxPool2d(kernel_size=2))

    return blocks


"""
    Define VGG Net based on original paper
    [https://arxiv.org/abs/1409.1556]
"""
class VGG(nn.Module):
    def __init__(self, in_c, n_cls, depths, _bn=False):
        super(VGG, self).__init__()

        block = basic_block
        blocks = []
        channels = [64, 128, 256, 512, 512]

        c = 0
        out_c = 64
        for depth in depths:
            out_c = channels[c]
            tmp_block = block(in_c, out_c, 3, depth, bn=_bn)
            in_c = out_c
            for b in tmp_block:
                blocks.append(b)
            c += 1

        self.feature = nn.Sequential(
            *blocks
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_cls)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x


"""
    Define VGG16, VGG16_BN, VGG19, VGG19_BN
"""
def vgg16(in_c, n_cls):
    return VGG(in_c, n_cls, [2, 2, 3, 3, 3], _bn=False)

def vgg19(in_c, n_cls):
    return VGG(in_c, n_cls, [2, 2, 4, 4, 4], _bn=False)

def vgg16_bn(in_c, n_cls):
    return VGG(in_c, n_cls, [2, 2, 3, 3, 3], _bn=True)

def vgg19_bn(in_c, n_cls):
    return VGG(in_c, n_cls, [2, 2, 4, 4, 4], _bn=True)


if __name__ == "__main__":
    from torchsummaryM import summary
    from torchvision.models import vgg

    official_model = vgg.vgg16()
    test_model = vgg16(3, 1000)

    inps = torch.zeros(1, 3, 224, 224)
    summary(official_model, inps)
    summary(test_model, inps)
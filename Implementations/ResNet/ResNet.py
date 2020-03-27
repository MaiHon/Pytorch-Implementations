import torch
import torch.nn as nn
from Blocks import BasicBlock, BottleBlock

"""
    Define ResNet based on
    https://arxiv.org/abs/1512.03385
"""
class ResNet(nn.Module):
    def __init__(self, in_c, n_cls, depths: list, block=BottleBlock):
        super(ResNet, self).__init__()
        self.in_c, self.n_cls = in_c, n_cls

        blocks = []
        channels = [64, 128, 256, 512]

        if block == BasicBlock:
            c = 1
            for _ in range(depths[0]): blocks.append(block(channels[0], channels[0]))
            for depth in depths[1:]:
                for n in range(depth):
                    if n == 0:
                        blocks.append(block(channels[c - 1], channels[c]))
                    else:
                        blocks.append(block(channels[c], channels[c]))
                c += 1
        else:
            c = 0
            in_c = 64
            out_c = 256
            for depth in depths:
                for n in range(depth):
                    blocks.append(block(in_c, out_c))
                    if n == 0 and c != 0: out_c *= 4
                    if n == 0: in_c = out_c
                out_c //= 2
                c += 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_c, 64, kernel_size=7, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            *blocks[:depths[0]]
        )
        self.conv3 = nn.Sequential(
            *blocks[depths[0]:sum(depths[:1])]
        )
        self.conv4 = nn.Sequential(
            *blocks[sum(depths[:1]):sum(depths[:2])]
        )
        self.conv5 = nn.Sequential(
            *blocks[sum(depths[:2]):sum(depths)]
        )

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512 * 4, n_cls) if block == BottleBlock else nn.Linear(512, n_cls)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.last(x)
        return x


def resnet18(in_c, n_cls):
    return ResNet(in_c, n_cls, [2, 2, 2, 2], BasicBlock)

def resnet34(in_c, n_cls):
    return ResNet(in_c, n_cls, [3, 4, 6, 3], BasicBlock)

def resnet50(in_c, n_cls):
    return ResNet(in_c, n_cls, [3, 4, 6, 3])

def resnet101(in_c, n_cls):
    return ResNet(in_c, n_cls, [3, 4, 23, 3])

def resnet152(in_c, n_cls):
    return ResNet(in_c, n_cls, [3, 8, 36, 3])


if __name__ == "__main__":
    from torchvision.models import resnet
    from torchsummaryM import summary


    test_model = resnet18(3, 1000)
    official_model = resnet.resnet18()

    summary(test_model, torch.zeros(1, 3, 224, 224))
    summary(official_model, torch.zeros(1, 3, 224, 224))
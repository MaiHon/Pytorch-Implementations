import torch.nn as nn

def activation_fun(activ):
    return nn.ModuleDict([
        ['relu', nn.ReLU(True)],
        ['leaky_relu', nn.LeakyReLU(0.02, True)],
        ['selu', nn.SELU(True)],
        ['none', nn.Identity()]
    ])[activ]

"""
    Define Basic Block 
"""
class BasicBlock(nn.Module):
    def __init__(self, in_f, out_f, a='relu'):
        super(BasicBlock, self).__init__()
        if in_f != out_f:
            self.conv1 = nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(out_f, out_f, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_f)
        self.bn2 = nn.BatchNorm2d(out_f)
        self.activ = activation_fun(a)

        if in_f != out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_f)
            )
        else:
            self.shortcut = False

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.shortcut:
            residual = self.shortcut(residual)

        x += residual
        x = self.activ(x)

        return x

"""
    Define Bottleneck Block
"""


class BottleBlock(nn.Module):
    def __init__(self, in_f, out_f, a='relu'):
        super(BottleBlock, self).__init__()

        if in_f == out_f: last = out_f; middle = in_f // 4
        if in_f > out_f: last = 4 * out_f; middle = out_f
        if in_f < out_f: last = out_f; middle = in_f

        if in_f > out_f:
            self.conv1 = nn.Conv2d(in_f, middle, kernel_size=1, stride=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_f, middle, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(middle, middle, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(middle, last, kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(middle)
        self.bn2 = nn.BatchNorm2d(middle)
        self.bn3 = nn.BatchNorm2d(last)
        self.activ = activation_fun(a)

        if in_f > out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, last, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(last)
            )
        elif in_f < out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, last, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(last)
            )
        else:
            self.shortcut = False

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activ(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.shortcut:
            residual = self.shortcut(residual)
        x += residual
        x = self.activ(x)
        return x
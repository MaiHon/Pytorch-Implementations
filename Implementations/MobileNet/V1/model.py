import torch
import torch.nn as nn


"""
    Depthwise Convolution Layer
        - consist of two layers, Depthwise & Pointwise
        - Depthwise convolution perform as a spatial filter
        - Pointwise convolution perform as a feature generater
        - Depthwise convolution produce similar performance to Normal convolution layers,
          but it does have lower parameters 
"""
class Deptwise_Conv(nn.Module):
    def __init__(self, alpha, k, s, in_c, out_c, img_size, last=False):
        super(Deptwise_Conv, self).__init__()
        """
            At groups=1
                - all inputs are convolved to all outputs.

            At groups=2
                - the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels,
                  and producing half the output channels, and both subsequently concatenated.

            At groups= in_channels
                - each input channel is convolved with its own set of filters, of size
        """
        in_c = int(in_c * alpha)
        out_c = int(out_c * alpha)

        last_size = img_size//(2**5)
        p = 1 if not last else (last_size-s+k)//2


        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c)
        self.bn1       = nn.BatchNorm2d(in_c)
        self.relu1     = nn.ReLU(True)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class MobileNet_v1(nn.Module):
    def __init__(self, alpha=1.0, img_size=224):
        super(MobileNet_v1, self).__init__()
        self.feature = self._build_features(alpha, img_size)
        self.classifier = self._build_classifier(alpha)

    def forward(self, x):
        out = self.feature(x)
        out = self.classifier(out)
        return out

    def _build_features(self, alpha, img_size):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=int(32*alpha), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU(True),

            Deptwise_Conv(alpha, 3, 1, 32, 64, img_size),
            Deptwise_Conv(alpha, 3, 2, 64, 128, img_size),
            Deptwise_Conv(alpha, 3, 1, 128, 128, img_size),
            Deptwise_Conv(alpha, 3, 2, 128, 256, img_size),
            Deptwise_Conv(alpha, 3, 1, 256, 256, img_size),
            Deptwise_Conv(alpha, 3, 2, 256, 512, img_size),

            Deptwise_Conv(alpha, 3, 1, 512, 512, img_size),
            Deptwise_Conv(alpha, 3, 1, 512, 512, img_size),
            Deptwise_Conv(alpha, 3, 1, 512, 512, img_size),
            Deptwise_Conv(alpha, 3, 1, 512, 512, img_size),
            Deptwise_Conv(alpha, 3, 1, 512, 512, img_size),

            Deptwise_Conv(alpha, 3, 2, 512, 1024, img_size),
            Deptwise_Conv(alpha, 3, 2, 1024, 1024, img_size, last=True),
        ]

        feature = nn.Sequential(
           *layers
        )
        return feature

    def _build_classifier(self, alpha):
        layers = [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=int(1024 * alpha), out_features=1000),
            nn.Softmax(dim=1)
        ]

        classifier = nn.Sequential(
            *layers
        )
        return classifier


if __name__ == "__main__":
    from torchsummaryM import summary

    model = MobileNet_v1(alpha=1.0)
    summary(model, torch.zeros(1, 3, 224, 224))

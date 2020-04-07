import torch
import torch.nn as nn

"""
    Inverted Residual Block   - 
"""
class Inverted_Residual_Block(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, t=6, alpha=1.0):
        super(Inverted_Residual_Block, self).__init__()

        """
            At groups=1
                - all inputs are convolved to all outputs.

            At groups=2
                - the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels,
                  and producing half the output channels, and both subsequently concatenated.

            At groups= in_channels
                - each input channel is convolved with its own set of filters, of size
        """

        in_C = int(in_c * alpha)
        middle_C = int(in_c * alpha * t)
        out_C = int(out_c * alpha)
        self.shortcut = True if s == 1 and in_C == out_C else False

        layers = []
        if t != 1:
            layers.extend([
                nn.Conv2d(in_C, middle_C, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(middle_C),
                nn.ReLU6(True)
            ])

        layers.extend([
            nn.Conv2d(middle_C, middle_C, kernel_size=k, stride=s, padding=(k-1)//2, groups=middle_C, bias=False),
            nn.ReLU6(True),
            nn.Conv2d(middle_C, out_C, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_C),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv(out)

        if self.shortcut:
            out = out + x
        return out



"""
    MobileNet_v2
"""
class MobileNet_v2(nn.Module):
    def __init__(self, alpha=1.0):
        super(MobileNet_v2, self).__init__()
        self.feature = self._build_features(alpha)
        self.classifier = self._build_classifier(alpha)

    def forward(self, x):
        out = self.feature(x)
        out = self.classifier(out)
        return out

    def _build_features(self, alpha):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=int(32*alpha), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU6(True),

            Inverted_Residual_Block(32, 16,   s=1, t=1, alpha=alpha),

            Inverted_Residual_Block(16, 24,   s=2, t=6, alpha=alpha),
            Inverted_Residual_Block(24, 24,   s=1, t=6, alpha=alpha),

            Inverted_Residual_Block(24, 32,   s=2, t=6, alpha=alpha),
            Inverted_Residual_Block(32, 32,   s=1, t=6, alpha=alpha),
            Inverted_Residual_Block(32, 32,   s=1, t=6, alpha=alpha),

            Inverted_Residual_Block(32, 64,   s=2, t=6, alpha=alpha),
            Inverted_Residual_Block(64, 64,   s=1, t=6, alpha=alpha),
            Inverted_Residual_Block(64, 64,   s=1, t=6, alpha=alpha),
            Inverted_Residual_Block(64, 64,   s=1, t=6, alpha=alpha),

            Inverted_Residual_Block(64, 96,   s=1, t=6, alpha=alpha),
            Inverted_Residual_Block(96, 96,   s=1, t=6, alpha=alpha),
            Inverted_Residual_Block(96, 96,   s=1, t=6, alpha=alpha),

            Inverted_Residual_Block(96, 160,  s=2, t=6, alpha=alpha),
            Inverted_Residual_Block(160, 160, s=1, t=6, alpha=alpha),
            Inverted_Residual_Block(160, 160, s=1, t=6, alpha=alpha),

            Inverted_Residual_Block(160, 320, s=1, t=6, alpha=alpha),

            nn.Conv2d(int(320*alpha), int(alpha*1280), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(alpha*1280)),
            nn.ReLU6(True)
        ]


        feature = nn.Sequential(
           *layers
        )
        return feature

    def _build_classifier(self, alpha):
        layers = [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=int(alpha*1280), out_features=1000),
        ]

        classifier = nn.Sequential(
            *layers
        )
        return classifier


if __name__ == "__main__":
    from torchsummaryM import summary
    from torchvision.models.mobilenet import MobileNetV2


    model = MobileNetV2()
    summary(model, torch.zeros(1, 3, 224, 224))

    model = MobileNet_v2(alpha=1.4)
    summary(model, torch.zeros(1, 3, 224, 224))

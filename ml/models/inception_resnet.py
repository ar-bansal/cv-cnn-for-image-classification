import torch.nn as nn
from .utils import ConvBlock, InceptionModule, SkipConnection, Model


class InceptionResNetStyleV1(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input shape = (B, 3, 32, 32)
        block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1), 
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1), 
        )
        self.block1 = SkipConnection(
            block1, 
            downsample=nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
            )
        # Output shape = (B, 32, 16, 16)

        block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1), 
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1), 
        )
        self.block2 = SkipConnection(
            block2, 
            downsample=nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            )
        # Output shape = (B, 64, 8, 8)

        inception3 = InceptionModule(
            in_channels=64, 
            f_1x1=8, 
            f_3x3_reduce=12, 
            f_3x3=16, 
            f_5x5_reduce=6, 
            f_5x5=12, 
            f_pp=12, 
        )
        self.inception3 = SkipConnection(
            inception3, 
            # Downsampling here uses a stride of 1 because the inception 
            # module only reduces the number of final filters, 
            # and not the spatial dimensions of the input tensors. 
            downsample=nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1) 
            )
        # Output shape = (B, 48, 8, 8)

        inception4 = InceptionModule(
            in_channels=48, 
            f_1x1=4, 
            f_3x3_reduce=8, 
            f_3x3=8, 
            f_5x5_reduce=8, 
            f_5x5=4, 
            f_pp=4, 
        )
        self.inception4 = SkipConnection(
            inception4, 
            # Downsampling here uses a stride of 1 because the inception 
            # module only reduces the number of final filters, 
            # and not the spatial dimensions of the input tensors.
            downsample=nn.Conv2d(48, 20, kernel_size=3, stride=1, padding=1)
            )
        # Output shape = (B, 20, 8, 8)

        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        # Output shape = (B, 20, 1, 1)

        self.flatten = nn.Flatten(1)
        # Output shape = (B, 20)

        self.fc6 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc6(x)

        return x
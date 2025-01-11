import torch.nn as nn
from .utils import ConvBlock, InceptionModule, SkipConnection, Model


class InceptionResNetStyleV1(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input shape = (B, 3, 32, 32)
        # self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        # Output shape = (B, 32, 32, 32)

        block2 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1), 
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1), 
        )
        self.block2 = SkipConnection(
            block2, 
            downsample=nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
            )
        # Output shape = (B, 32, 16, 16)

        block3 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1), 
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1), 
        )
        self.block3 = SkipConnection(
            block3, 
            downsample=nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            )
        # Output shape = (B, 64, 8, 8)

        inception4 = InceptionModule(
            in_channels=64, 
            f_1x1=8, 
            f_3x3_reduce=12, 
            f_3x3=16, 
            f_5x5_reduce=6, 
            f_5x5=12, 
            f_pp=12, 
        )
        self.inception4 = SkipConnection(
            inception4, 
            # Downsampling here uses a stride of 1 because the inception 
            # module only reduces the number of final filters, 
            # and not the spatial dimensions of the input tensors. 
            downsample=nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1) 
            )
        # Output shape = (B, 48, 8, 8)

        inception5 = InceptionModule(
            in_channels=48, 
            f_1x1=4, 
            f_3x3_reduce=8, 
            f_3x3=8, 
            f_5x5_reduce=8, 
            f_5x5=4, 
            f_pp=4, 
        )
        self.inception5 = SkipConnection(
            inception5, 
            # Downsampling here uses a stride of 1 because the inception 
            # module only reduces the number of final filters, 
            # and not the spatial dimensions of the input tensors.
            downsample=nn.Conv2d(48, 20, kernel_size=3, stride=1, padding=1)
            )
        # Output shape = (B, 20, 8, 8)

        self.pool6 = nn.AdaptiveAvgPool2d((1, 1))
        # Output shape = (B, 20, 1, 1)

        self.flatten = nn.Flatten(1)
        # Output shape = (B, 20)

        self.fc7 = nn.Linear(20, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # print(x.shape)
        x = self.block2(x)
        print(x.shape)
        x = self.block3(x)
        print(x.shape)
        x = self.inception4(x)
        print(x.shape)
        x = self.inception5(x)
        print(x.shape)
        x = self.pool6(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc7(x)
        print(x.shape)

        return x
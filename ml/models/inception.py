import torch.nn as nn
from .utils import ConvBlock, InceptionModule, Model
    

class InceptionStyleV1(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input shape = (B, 3, 32, 32)
        self.block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1), 
            ConvBlock(32, 32, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 32, 16, 16)

        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, padding=1), 
            ConvBlock(64, 64, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 64, 8, 8)

        self.inception3 = InceptionModule(
            in_channels=64, 
            f_1x1=4, 
            f_3x3_reduce=8, 
            f_3x3=8, 
            f_5x5_reduce=8, 
            f_5x5=4, 
            f_pp=4, 
        )
        # Output shape = (B, 20, 8, 8)

        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # Output shape = (B, 20, 1, 1)

        self.flatten = nn.Flatten(1)
        # Output shape = (B, 20)

        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.inception3(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
    

class InceptionStyleV2(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input shape = (B, 3, 32, 32)
        self.block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1), 
            ConvBlock(32, 32, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 32, 16, 16)

        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, padding=1), 
            ConvBlock(64, 64, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 64, 8, 8)

        self.inception3 = InceptionModule(
            in_channels=64, 
            f_1x1=8, 
            f_3x3_reduce=12, 
            f_3x3=16, 
            f_5x5_reduce=6, 
            f_5x5=12, 
            f_pp=12, 
        )
        # Output shape = (B, 48, 8, 8)

        self.inception4 = InceptionModule(
            in_channels=48, 
            f_1x1=4, 
            f_3x3_reduce=8, 
            f_3x3=8, 
            f_5x5_reduce=8, 
            f_5x5=4, 
            f_pp=4, 
        )
        # Output shape = (B, 20, 8, 8)

        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # Output shape = (B, 20, 1, 1)

        self.flatten = nn.Flatten(1)
        # Output shape = (B, 20)

        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
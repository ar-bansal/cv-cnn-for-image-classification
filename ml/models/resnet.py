import torch.nn as nn
import torch.optim as optim
from .utils import Model, ConvBlock, SkipConnection


class ResNetStyleV1(Model):
    def __init__(self):
        super(ResNetStyleV1, self).__init__()

        # Input shape = (B, 3, 32, 32)
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        #  Output shape = (B, 32, 32, 32)

        self.res1 = SkipConnection(
            nn.Sequential(
                ConvBlock(32, 32, kernel_size=3, stride=1, padding=1), 
                ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)
            )
        )
        # Output shape = (B, 32, 32, 32)

        self.res2 = SkipConnection(
            nn.Sequential(
                ConvBlock(32, 64, kernel_size=3, stride=2, padding=1), 
                ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
            ), 
            downsample=nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        # Output shape = (B, 64, 16, 16)

        self.res3 = SkipConnection(
            nn.Sequential(
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1), 
                ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
            ), 
            downsample=nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        # Output shape = (B, 128, 8, 8)

        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # Output shape = (B, 128, 1, 1)

        self.flatten = nn.Flatten(1)
        # Output shape = (B, 128)

        self.fc5 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc5(x)

        return x
    

class ResNetDropoutV1(ResNetStyleV1):
    def __init__(self):
        super(ResNetDropoutV1, self).__init__()
        
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc5(x)
        x = self.dropout(x)

        return x


class ResNetWDecayV1(ResNetStyleV1):
    def __init__(self):
        super(ResNetWDecayV1, self).__init__()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-1)
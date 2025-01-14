import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .utils import Model, ConvBlock, SkipConnection


class ResNetStyleV1(Model):
    def __init__(self):
        super(ResNetStyleV1, self).__init__()

        self.relu = nn.ReLU()
        # Input shape = (B, 3, 32, 32)
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        #  Output shape = (B, 32, 32, 32)

        self.res2 = SkipConnection(
            nn.Sequential(
                ConvBlock(32, 32, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            )
        )
        # Output shape = (B, 32, 32, 32)

        self.res3 = SkipConnection(
            nn.Sequential(
                ConvBlock(32, 64, kernel_size=3, stride=2, padding=1), 
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            ), 
            downsample=nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        # Output shape = (B, 64, 16, 16)

        self.res4 = SkipConnection(
            nn.Sequential(
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1), 
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            ), 
            downsample=nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        # Output shape = (B, 128, 8, 8)

        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        # Output shape = (B, 128, 1, 1)

        self.flatten = nn.Flatten(1)
        # Output shape = (B, 128)

        self.fc6 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)

        x = self.res2(x)
        x = self.relu(x)

        x = self.res3(x)
        x = self.relu(x)

        x = self.res4(x)
        x = self.relu(x)

        x = self.pool5(x)
        x = self.flatten(x)

        x = self.fc6(x)

        return x
    

class ResNetDropoutV1(ResNetStyleV1):
    def __init__(self):
        super(ResNetDropoutV1, self).__init__()
        
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.conv1(x)

        x = self.res2(x)
        x = self.relu(x)

        x = self.res3(x)
        x = self.relu(x)

        x = self.res4(x)
        x = self.relu(x)

        x = self.pool5(x)
        x = self.flatten(x)

        x = self.fc6(x)
        x = self.dropout(x)

        return x
    

class ResNetDropoutV2(ResNetDropoutV1):
    """
    Uses dropout after the GAP layer, before the fully connected layer. 
    """
    def __init__(self):
        super(ResNetDropoutV2, self).__init__()

    def forward(self, x):
        x = self.conv1(x)

        x = self.res2(x)
        x = self.relu(x)

        x = self.res3(x)
        x = self.relu(x)

        x = self.res4(x)
        x = self.relu(x)

        x = self.pool5(x)
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.fc6(x)

        return x


class ResNetDropoutV3(ResNetDropoutV1):
    """
    Uses dropout before the GAP layer. 
    """
    def __init__(self):
        super(ResNetDropoutV3, self).__init__()

        self.dropout2d = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)

        x = self.res2(x)
        x = self.relu(x)

        x = self.res3(x)
        x = self.relu(x)

        x = self.res4(x)
        x = self.relu(x)

        x = self.dropout2d(x)
        x = self.pool5(x)
        x = self.flatten(x)

        x = self.fc6(x)

        return x


class ResNetDropoutV4(ResNetStyleV1):
    """
    Use 2-D dropout in all skip-connected blocks. The dropout is placed
    after the conv2d layer in the residual branch. 
    """
    def __init__(self):
        super().__init__()

        self.res2 = SkipConnection(
            nn.Sequential(
                ConvBlock(32, 32, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), 
                nn.Dropout2d(0.1), 
            )
        )
        # Output shape = (B, 32, 32, 32)

        self.res3 = SkipConnection(
            nn.Sequential(
                ConvBlock(32, 64, kernel_size=3, stride=2, padding=1), 
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                nn.Dropout2d(0.1), 
            ), 
            downsample=nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        # Output shape = (B, 64, 16, 16)

        self.res4 = SkipConnection(
            nn.Sequential(
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1), 
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                nn.Dropout2d(0.1), 
            ), 
            downsample=nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        # Output shape = (B, 128, 8, 8)


class ResNetWDecayV1(ResNetStyleV1):
    def __init__(self):
        super(ResNetWDecayV1, self).__init__()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-1)
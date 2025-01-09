import torch.nn as nn
from .utils import ConvBlock, Model


class VGGStyleV1(Model):
    """
    Uses 1 VGG block.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input shape = (B, 3, 32, 32)
        self.block = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1), 
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 32, 16, 16)

        self.fc = nn.Linear(32 * 16 * 16, 10)


    def forward(self, x):
        x = self.block(x)
        # Flatten the processed features
        x = x.flatten(1)
        return self.fc(x)
    

class VGGStyleV2(Model):
    """
    Uses 2 VGG blocks. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Input shape = (B, 3, 32, 32)
        self.block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1), 
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 32, 16, 16)

        # Input shape = (B, 3, 32, 32)
        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1), 
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 64, 8, 8)
        
        self.flatten = nn.Flatten()
        # Output shape = (B, 4096)

        self.fc = nn.Linear(4096, 10)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class VGGStyleV3(Model):
    """
    Uses 3 VGG blocks. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Input shape = (B, 3, 32, 32)
        self.block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1), 
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 32, 16, 16)

        # Input shape = (B, 3, 32, 32)
        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1), 
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 64, 8, 8)

        self.block3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1), 
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape = (B, 128, 4, 4)

        self.flatten = nn.Flatten(1)
        # Output shape = (B, 2048)

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import pytorch_lightning as L


class VGGStyleV1(L.LightningModule):
    """
    Uses 1 VGG block.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # in features = image size * number of filters
        self.fc = nn.Linear(16 * 16 * 32, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        x = self.block(x)
        # Flatten the processed features
        x = x.flatten(1)
        return self.fc(x)
    
    def training_step(self, batch, batch_idx):
        # return super().training_step(*args, **kwargs)
        x, y = batch
        outputs = self(x)

        loss = self.loss_fn(outputs, y)
        train_accuracy = self.accuracy(outputs, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True)

        return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)

    #     y_preds = torch.argmax(logits)

    #     return y_preds.cpu().numpy(), y.cpu().numpy()
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)


class VGGStyleV2(L.LightningModule):
    """
    Uses 2 VGG blocks. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # in features = image size * number of filters
        self.fc = nn.Linear(8 * 8 * 64, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        # Flatten the processed features
        x = x.flatten(1)
        return self.fc(x)
    
    def training_step(self, batch, batch_idx):
        # return super().training_step(*args, **kwargs)
        x, y = batch
        outputs = self(x)

        loss = self.loss_fn(outputs, y)
        train_accuracy = self.accuracy(outputs, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True)

        return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)

    #     y_preds = torch.argmax(logits)

    #     return y_preds.cpu().numpy(), y.cpu().numpy()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)



class VGGStyleV3(L.LightningModule):
    """
    Uses 3 VGG blocks. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # in features = image size * number of filters
        self.fc = nn.Linear(4 * 4 * 128, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Flatten the processed features
        x = x.flatten(1)
        return self.fc(x)
    
    def training_step(self, batch, batch_idx):
        # return super().training_step(*args, **kwargs)
        x, y = batch
        outputs = self(x)

        loss = self.loss_fn(outputs, y)
        train_accuracy = self.accuracy(outputs, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True)

        return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)

    #     y_preds = torch.argmax(logits)

    #     return y_preds.cpu().numpy(), y.cpu().numpy()
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)

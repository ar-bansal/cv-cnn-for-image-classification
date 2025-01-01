import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import pytorch_lightning as L


class VGGBase(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = Accuracy("multiclass", num_classes=10)

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)

        train_loss = self.loss_fn(outputs, y)
        train_accuracy = self.accuracy_fn(outputs, y)

        self.log("train_loss", train_loss, on_epoch=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)

        val_loss = self.loss_fn(outputs, y)
        val_accuracy = self.accuracy_fn(outputs, y)

        self.log("val_loss", val_loss, on_epoch=True)
        self.log("val_accuracy", val_accuracy, on_epoch=True)
    

    def test_step(self, batch, batch_idx):
        images, true_labels = batch
        logits = self(images)

        pred_labels = torch.argmax(logits, dim=1)

        test_accuracy = self.accuracy_fn(pred_labels, true_labels)

        self.log("test_accuracy", test_accuracy, on_epoch=True)
        return test_accuracy
    

    def predict_step(self, batch, batch_idx):
        logits = self(batch)

        pred_labels = torch.argmax(logits, dim=1)
        return pred_labels
    

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)


class VGGStyleV1(VGGBase):
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


    def forward(self, x):
        x = self.block(x)
        # Flatten the processed features
        x = x.flatten(1)
        return self.fc(x)
    

class VGGStyleV2(VGGBase):
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


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        # Flatten the processed features
        x = x.flatten(1)
        return self.fc(x)


class VGGStyleV3(VGGBase):
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

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Flatten the processed features
        x = x.flatten(1)
        return self.fc(x)

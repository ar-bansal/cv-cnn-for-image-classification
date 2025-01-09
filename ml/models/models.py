# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchmetrics import Accuracy
# import pytorch_lightning as L
# from utils import ConvBlock, InceptionModule, ResidualBlock


# # class VGGBase(L.LightningModule):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.loss_fn = nn.CrossEntropyLoss()
# #         self.accuracy_fn = Accuracy("multiclass", num_classes=10)

    
# #     def training_step(self, batch, batch_idx):
# #         x, y = batch
# #         outputs = self(x)

# #         train_loss = self.loss_fn(outputs, y)
# #         train_accuracy = self.accuracy_fn(outputs, y)

# #         self.log("train_loss", train_loss, on_epoch=True)
# #         self.log("train_accuracy", train_accuracy, on_epoch=True)

# #         return train_loss
    

# #     def validation_step(self, batch, batch_idx):
# #         x, y = batch
# #         outputs = self(x)

# #         val_loss = self.loss_fn(outputs, y)
# #         val_accuracy = self.accuracy_fn(outputs, y)

# #         self.log("val_loss", val_loss, on_epoch=True)
# #         self.log("val_accuracy", val_accuracy, on_epoch=True)
    

# #     def test_step(self, batch, batch_idx):
# #         images, true_labels = batch
# #         logits = self(images)

# #         pred_labels = torch.argmax(logits, dim=1)

# #         test_accuracy = self.accuracy_fn(pred_labels, true_labels)

# #         self.log("test_accuracy", test_accuracy, on_epoch=True)
# #         return test_accuracy
    

# #     def predict_step(self, batch, batch_idx):
# #         logits = self(batch)

# #         pred_labels = torch.argmax(logits, dim=1)
# #         return pred_labels
    

# #     def configure_optimizers(self):
# #         return optim.AdamW(self.parameters(), lr=1e-3)


# class VGGStyleV1(VGGBase):
#     """
#     Uses 1 VGG block.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.block = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.Conv2d(32, 32, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # in features = image size * number of filters
#         self.fc = nn.Linear(16 * 16 * 32, 10)


#     def forward(self, x):
#         x = self.block(x)
#         # Flatten the processed features
#         x = x.flatten(1)
#         return self.fc(x)
    

# class VGGStyleV2(VGGBase):
#     """
#     Uses 2 VGG blocks. 
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.Conv2d(32, 32, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.Conv2d(64, 64, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # in features = image size * number of filters
#         self.fc = nn.Linear(8 * 8 * 64, 10)


#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         # Flatten the processed features
#         x = x.flatten(1)
#         return self.fc(x)


# class VGGStyleV3(VGGBase):
#     """
#     Uses 3 VGG blocks. 
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.Conv2d(32, 32, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.Conv2d(64, 64, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         self.block3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.Conv2d(128, 128, kernel_size=3, padding=1), 
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # in features = image size * number of filters
#         self.fc = nn.Linear(4 * 4 * 128, 10)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         # Flatten the processed features
#         x = x.flatten(1)
#         return self.fc(x)


# # class ConvBlock(nn.Module):
# #     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
# #         super(ConvBlock, self).__init__()

# #         self.conv = nn.Conv2d(
# #             in_channels, 
# #             out_channels, 
# #             kernel_size, 
# #             stride, 
# #             padding
# #         )
# #         self.act = nn.ReLU()


# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.act(x)
# #         return x



# # class InceptionModule(nn.Module):
# #     def __init__(self, in_channels, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp):
# #         super(InceptionModule, self).__init__()

# #         self.branch1 = nn.Sequential(
# #             ConvBlock(in_channels, f_1x1, kernel_size=1, stride=1, padding=0)
# #         )

# #         self.branch2 = nn.Sequential(
# #             ConvBlock(in_channels, f_3x3_reduce, kernel_size=1, stride=1, padding=0), 
# #             ConvBlock(f_3x3_reduce, f_3x3, kernel_size=3, stride=1, padding=1) 
# #         )

# #         self.branch3 = nn.Sequential(
# #             ConvBlock(in_channels, f_5x5_reduce, kernel_size=1, stride=1, padding=0), 
# #             ConvBlock(f_5x5_reduce, f_5x5, kernel_size=5, stride=1, padding=2) 
# #         )

# #         self.branch4 = nn.Sequential(
# #             nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True), 
# #             ConvBlock(in_channels, f_pp, kernel_size=1, stride=1, padding=0)
# #         )

    
# #     def forward(self, x):
# #         branch1 = self.branch1(x)
# #         branch2 = self.branch2(x)
# #         branch3 = self.branch3(x)
# #         branch4 = self.branch4(x)

# #         return torch.cat([branch1, branch2, branch3, branch4], dim=1)
    

# class InceptionBase(L.LightningModule):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.accuracy_fn = Accuracy("multiclass", num_classes=10)

    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         outputs = self(x)

#         train_loss = self.loss_fn(outputs, y)
#         train_accuracy = self.accuracy_fn(outputs, y)

#         self.log("train_loss", train_loss, on_epoch=True)
#         self.log("train_accuracy", train_accuracy, on_epoch=True)

#         return train_loss
    

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         outputs = self(x)

#         val_loss = self.loss_fn(outputs, y)
#         val_accuracy = self.accuracy_fn(outputs, y)

#         self.log("val_loss", val_loss, on_epoch=True)
#         self.log("val_accuracy", val_accuracy, on_epoch=True)
    

#     def test_step(self, batch, batch_idx):
#         images, true_labels = batch
#         logits = self(images)

#         pred_labels = torch.argmax(logits, dim=1)

#         test_accuracy = self.accuracy_fn(pred_labels, true_labels)

#         self.log("test_accuracy", test_accuracy, on_epoch=True)
#         return test_accuracy
    

#     def predict_step(self, batch, batch_idx):
#         logits = self(batch)

#         pred_labels = torch.argmax(logits, dim=1)
#         return pred_labels
    

#     def configure_optimizers(self):
#         return optim.AdamW(self.parameters(), lr=1e-3)
    

# class InceptionStyleV1(InceptionBase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Input shape = (B, 3, 32, 32)
#         self.block1 = nn.Sequential(
#             ConvBlock(3, 32, kernel_size=3, padding=1), 
#             ConvBlock(32, 32, kernel_size=3, padding=1), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Output shape = (B, 32, 16, 16)

#         self.block2 = nn.Sequential(
#             ConvBlock(32, 64, kernel_size=3, padding=1), 
#             ConvBlock(64, 64, kernel_size=3, padding=1), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Output shape = (B, 64, 8, 8)

#         self.inception3 = InceptionModule(
#             in_channels=64, 
#             f_1x1=4, 
#             f_3x3_reduce=8, 
#             f_3x3=8, 
#             f_5x5_reduce=8, 
#             f_5x5=4, 
#             f_pp=4, 
#         )
#         # Output shape = (B, 20, 8, 8)

#         self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
#         # Output shape = (B, 20, 1, 1)

#         self.fc = nn.Linear(20, 10)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.inception3(x)
#         x = self.pool4(x)
#         x = x.flatten(1)
#         return self.fc(x)
    

# class InceptionStyleV2(InceptionBase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Input shape = (B, 3, 32, 32)
#         self.block1 = nn.Sequential(
#             ConvBlock(3, 32, kernel_size=3, padding=1), 
#             ConvBlock(32, 32, kernel_size=3, padding=1), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Output shape = (B, 32, 16, 16)

#         self.block2 = nn.Sequential(
#             ConvBlock(32, 64, kernel_size=3, padding=1), 
#             ConvBlock(64, 64, kernel_size=3, padding=1), 
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # Output shape = (B, 64, 8, 8)

#         self.inception3 = InceptionModule(
#             in_channels=64, 
#             f_1x1=8, 
#             f_3x3_reduce=12, 
#             f_3x3=16, 
#             f_5x5_reduce=6, 
#             f_5x5=12, 
#             f_pp=12, 
#         )
#         # Output shape = (B, 48, 8, 8)

#         self.inception4 = InceptionModule(
#             in_channels=48, 
#             f_1x1=4, 
#             f_3x3_reduce=8, 
#             f_3x3=8, 
#             f_5x5_reduce=8, 
#             f_5x5=4, 
#             f_pp=4, 
#         )
#         # Output shape = (B, 20, 8, 8)

#         self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
#         # Output shape = (B, 20, 1, 1)

#         self.fc = nn.Linear(20, 10)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.inception3(x)
#         x = self.inception4(x)
#         x = self.pool4(x)
#         x = x.flatten(1)
#         return self.fc(x)
    

# # class ResidualBlock(nn.Module):
# #     def __init__(self, in_channels, out_channels, stride, downsample=None):
# #         super(ResidualBlock, self).__init__()

# #         self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
# #         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
# #         self.relu = nn.ReLU()
# #         self.downsample = downsample

# #     def forward(self, x):
# #         identity = x
# #         if self.downsample:
# #             identity = self.downsample(x)

# #         out = self.conv1(x)
# #         out = self.conv2(out)
# #         out += identity
# #         out = self.relu(out)

# #         return out


# class ResNetV1(InceptionBase):
#     def __init__(self):
#         super(ResNetV1, self).__init__()

#         # Input shape = (B, 3, 32, 32)
#         self.res1 = ResidualBlock(3, 32, stride=1)
#         # Output shape = (B, 32, 32, 32)

#         downsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
#         self.res2 = ResidualBlock(32, 64, stride=1, downsample=downsample1)
#         # Output shape = (B, 64, 16, 16)

#         downsample2 = nn.Conv2d(128, 128, kernel_size=3, stride=3)
#         self.res3 = ResidualBlock(64, 128, stride=1, downsample=downsample2)
#         # Output shape = (B, 128, 8, 8)

#         self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
#         # Output shape = (B, 128, 1, 1)

#         # Input shape = (B, 128) [after flattening]
#         self.fc5 = nn.Linear(128, 10)
#         # Output shape = (B, 10) 


#     def forward(self, x):
#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.res3(x)
#         x = self.pool4(x)
#         x = x.flatten(1)
#         x = self.fc5(x)

#         return x
    

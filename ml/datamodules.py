import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset, val_size, batch_size, train_transform=None, test_transform=None, num_workers=None, data_dir="data", random_state=0):
        super().__init__()

        self.dataset = dataset
        self.val_size = val_size
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_workers = num_workers
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.random_state = random_state
        self.pin_memory = True if torch.cuda.is_available() else False

    def prepare_data(self):
        # Download train data
        self.dataset(
            root=os.path.join(self.data_dir, "train"), 
            train=True, 
            download=True
        )

        # Download test data
        self.dataset(
            root=os.path.join(self.data_dir, "test"), 
            train=False, 
            download=True
        )

    def setup(self, stage=None):
        train_set = self.dataset(
            root=os.path.join(self.data_dir, "train"), 
            train=True, 
            transform=self.train_transform, 
            download=True
        )
        val_set = self.dataset(
            root=os.path.join(self.data_dir, "train"), 
            train=True, 
            transform=self.test_transform, 
            download=True
        )
        targets = train_set.targets

        train_indices, val_indices = train_test_split(
            range(len(targets)), 
            test_size=self.val_size, 
            stratify=targets, 
            random_state=self.random_state
        )

        self.train_dataset = Subset(train_set, train_indices)
        self.val_dataset = Subset(val_set, val_indices)

        self.test_dataset = self.dataset(
            root=os.path.join(self.data_dir, "test"), 
            train=False, 
            transform=self.test_transform, 
            download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, val_size, batch_size, train_transform=None, test_transform=None, data_dir="data", random_state=0):
        dataset = CIFAR10
        num_workers = os.cpu_count()
        
        default_transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        if not train_transform:
            train_transform = default_transform
        if not test_transform:
            test_transform = default_transform
            
        super().__init__(dataset, val_size, batch_size, train_transform, test_transform, num_workers, data_dir, random_state)

    def predict_dataloader(self):
        class InferenceDataset(Dataset):
            def __init__(self, data, transform=None):
                self.data = data
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                image = self.data[idx]
                if self.transform:
                    image = self.transform(image)
                return image

        # Create the inference dataset using test_dataset.data
        inference_dataset = InferenceDataset(self.test_dataset.data, transform=self.test_transform)

        return DataLoader(
            inference_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
    

class CIFAR10TestingModule(CIFAR10DataModule):
    """
    Data module used for testing models on a very small set before 
    training on a GPU.
    """

    def __init__(self, val_size=0.15, batch_size=10, train_transform=None, test_transform=None, data_dir="data", random_state=0):
        super().__init__(val_size, batch_size, train_transform, test_transform, data_dir, random_state)

    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = Subset(self.train_dataset, range(200))
        self.val_dataset = Subset(self.val_dataset, range(200))
        self.test_dataset = Subset(self.test_dataset, range(20))
import os
import mlflow
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from ml.models import ResNetStyleV1
from ml.pipelines import run_pipeline
from mlops.ml_logging import get_tracking_uri


def main():
    MODEL = ResNetStyleV1()
    EXPERIMENT_NAME = "resnet-style-cnn"
    RANDOM_STATE = 0
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    NUM_WORKERS = os.cpu_count()
    TEST_SIZE = 0.05
    PIN_MEMORY = True if torch.cuda.is_available() else False

    tracking_server_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_server_uri)

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = CIFAR10(root="./data/train", train=True, transform=transform, download=True)
    targets = dataset.targets

    # Split the dataset into train and validation sets. 
    train_indices, val_indices = train_test_split(
        range(len(targets)), 
        test_size=TEST_SIZE, 
        stratify=targets, 
        random_state=RANDOM_STATE
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = CIFAR10(root="./data/test", train=False, transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY

    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )

    class_labels = [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
   
    run_pipeline(
        MODEL, 
        NUM_EPOCHS, 
        train_loader, 
        val_loader, 
        test_loader, 
        class_labels, 
        experiment_name=EXPERIMENT_NAME
    )


if __name__ == "__main__":
    main()
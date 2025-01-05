import requests
import mlflow
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from ml.models import * 
from ml.pipelines import run_pipeline


def main():
    MLFLOW_TRACKING_URI = "http://localhost:5001/"
    try:
        requests.get(MLFLOW_TRACKING_URI)
    except:
        raise requests.exceptions.ConnectionError("Unable to reach MLOps.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    RANDOM_STATE = 0
    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    NUM_WORKERS = 2
    NUM_SAMPLES = 200
    PIN_MEMORY = True if torch.cuda.is_available() else False

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
        test_size=0.2, 
        stratify=targets, 
        random_state=RANDOM_STATE
    )

    train_dataset = Subset(dataset, train_indices[:NUM_SAMPLES])
    val_dataset = Subset(dataset, val_indices[:NUM_SAMPLES])
    test_dataset = Subset(
        CIFAR10(root="./data/test", train=False, transform=transform, download=True), 
        range(NUM_SAMPLES)
    )

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
        InceptionStyleV2(), 
        NUM_EPOCHS, 
        train_loader, 
        val_loader, 
        test_loader, 
        class_labels, 
        experiment_name="inception-style-cnn"
    )


if __name__ == "__main__":
    main()
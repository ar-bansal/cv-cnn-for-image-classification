import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning import Trainer
from mlops.ml_logging import get_tracking_uri, log_pytorch


__all__ = ["run_pipeline"]


def create_trainer(n_epochs: int=None) -> Trainer:
    trainer = Trainer(
            max_epochs=n_epochs, 
            logger=False, 
            enable_checkpointing=False
        )
    return trainer


def train(model, train_dataloader, val_dataloader, n_epochs):
    trainer = create_trainer(n_epochs)
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
        )

    return trainer.model


def evaluate(model, test_dataloader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, targets in test_dataloader:
            logits = model(x)
            y_pred = torch.argmax(logits, 1)
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_targets)


def get_metrics(predictions, targets, class_labels=None):
    predicted_classes = np.array(class_labels)[predictions]
    target_classes = np.array(class_labels)[targets]

    cls_report = classification_report(
        target_classes, 
        predicted_classes, 
        target_names=class_labels, 
        output_dict=True
    )

    report_df = pd.DataFrame(cls_report).transpose().round(4)

    true_cols = [f"true_{label}" for label in class_labels]
    pred_idxs = [f"pred_{label}" for label in class_labels]

    cm = confusion_matrix(
        target_classes, 
        predicted_classes, 
        labels=class_labels
    ).T

    cm_df = pd.DataFrame(cm, columns=true_cols, index=pred_idxs)

    return {
        "test_accuracy": round(accuracy_score(target_classes, predicted_classes), 4), 
        "test_classification_report": report_df, 
        "test_confusion_matrix": cm_df
    }


@log_pytorch(save_graph=True)
def train_and_evaluate(model, n_epochs, train_loader, val_loader, test_loader, class_labels, experiment_name, input_shape):
    model = train(model, train_loader, val_loader, n_epochs)

    preds, targets = evaluate(model, test_loader)

    metrics = get_metrics(preds, targets, class_labels=class_labels)

    return model, metrics


def run_pipeline(model, batch_size, num_epochs, num_workers, val_size, experiment_name, random_state):
    PIN_MEMORY = True if torch.cuda.is_available() else False
    DATA_DIR = os.path.join(os.getcwd(), "data")
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
    TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
    INPUT_SHAPE = (batch_size, 3, 32, 32)

    tracking_server_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_server_uri)

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = CIFAR10(root=TRAIN_DATA_DIR, train=True, transform=transform, download=True)
    targets = dataset.targets

    # Split the dataset into train and validation sets. 
    train_indices, val_indices = train_test_split(
        range(len(targets)), 
        test_size=val_size, 
        stratify=targets, 
        random_state=random_state
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = CIFAR10(root=TEST_DATA_DIR, train=False, transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=PIN_MEMORY

    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  
        num_workers=num_workers, 
        pin_memory=PIN_MEMORY
    )

    class_labels = [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
   
    train_and_evaluate(
        model, 
        num_epochs, 
        train_loader, 
        val_loader, 
        test_loader, 
        class_labels, 
        experiment_name=experiment_name, 
        input_shape=INPUT_SHAPE
    )
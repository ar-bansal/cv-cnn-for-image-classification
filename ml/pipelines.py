import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from mlops.ml_logging import log_pytorch


__all__ = ["run_pipeline"]


def create_trainer(n_epochs: int=None) -> Trainer:
    trainer = Trainer(
            accelerator="gpu", 
            devices=1, 
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


@log_pytorch(logging_kwargs={"log_every_n_epoch": 1})
def run_pipeline(model, n_epochs, train_loader, val_loader, test_loader, class_labels, experiment_name):
    model = train(model, train_loader, val_loader, n_epochs)

    preds, targets = evaluate(model, test_loader)

    metrics = get_metrics(preds, targets, class_labels=class_labels)

    return model, metrics

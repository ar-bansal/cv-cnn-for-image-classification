import numpy as np
import pandas as pd
import torch
import pytorch_lightning as L
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from mlops.ml_logging import log_pytorch


__all__ = ["run_pipeline"]

def train(model, data_loader, n_epochs):
    trainer = L.Trainer(
        accelerator="auto", 
        strategy="auto", 
        devices="auto", 
        max_epochs=n_epochs, 
        logger=False, 
        enable_checkpointing=False
    )
    trainer.fit(model=model, train_dataloaders=data_loader)

    return trainer

def evaluate(model, data_loader):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, targets in data_loader:
            logits = model(x)
            y_pred = torch.argmax(logits, 1)
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_targets)

def get_metrics(predictions, targets, class_labels=None):
    cls_report = classification_report(
        targets, 
        predictions, 
        target_names=class_labels, 
        output_dict=True
    )

    report_df = pd.DataFrame(cls_report).transpose().round(4)

    true_cols = [f"true_{label}" for label in class_labels]
    pred_idxs = [f"pred_{label}" for label in class_labels]

    cm = confusion_matrix(
        targets, 
        predictions, 
        labels=class_labels
    ).T

    cm_df = pd.DataFrame(cm, columns=true_cols, index=pred_idxs)

    return {
        "valid_accuracy": round(accuracy_score(targets, predictions), 4), 
        "valid_classification_report": report_df, 
        "valid_confusion_matrix": cm_df
    }


@log_pytorch(logging_kwargs={"log_every_n_epoch": 1})
def run_pipeline(model, n_epochs, train_data_loader, test_data_loader, class_labels, experiment_name):
    model = train(model, train_data_loader, n_epochs)

    preds, targets = evaluate(model, test_data_loader)

    metrics = get_metrics(preds, targets, class_labels=class_labels)

    return model, metrics

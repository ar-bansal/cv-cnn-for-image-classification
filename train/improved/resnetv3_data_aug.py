import mlflow
from torchvision import transforms
from mlops.ml_logging import get_tracking_uri
from ml.models import ResNetV3
from ml.pipelines import run_pipeline
from ml.datamodules import CIFAR10DataModule


if __name__ == "__main__":
    MODEL = ResNetV3()
    VAL_SIZE = 0.05
    BATCH_SIZE = 64
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    NUM_EPOCHS = 200
    EXPERIMENT_NAME = "resnet-style-cnn"
    
    DATAMODULE = CIFAR10DataModule(
        val_size=VAL_SIZE, 
        batch_size=BATCH_SIZE, 
        train_transform=TRAIN_TRANSFORM
    )

    tracking_server_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_server_uri)

    run_pipeline(
        model=MODEL, 
        datamodule=DATAMODULE, 
        num_epochs=NUM_EPOCHS,
        experiment_name=EXPERIMENT_NAME, 
    )
    
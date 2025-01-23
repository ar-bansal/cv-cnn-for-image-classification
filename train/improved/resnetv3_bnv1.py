import mlflow
from mlops.ml_logging import get_tracking_uri
from ml.models import ResNetV3_BNV1
from ml.pipelines import run_pipeline
from ml.datamodules import CIFAR10DataModule


if __name__ == "__main__":
    MODEL = ResNetV3_BNV1()
    VAL_SIZE = 0.05
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    EXPERIMENT_NAME = "resnet-style-cnn"
    
    DATAMODULE = CIFAR10DataModule(
        val_size=VAL_SIZE, 
        batch_size=BATCH_SIZE, 
    )

    tracking_server_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_server_uri)

    run_pipeline(
        model=MODEL, 
        datamodule=DATAMODULE, 
        num_epochs=NUM_EPOCHS,
        experiment_name=EXPERIMENT_NAME, 
    )
    
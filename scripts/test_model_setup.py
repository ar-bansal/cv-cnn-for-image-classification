import requests
import mlflow
import time
from mlops.server.operations import Server
from ml.models import ResNetV3_WInit
from ml.pipelines import run_pipeline
from ml.datamodules import CIFAR10DataModule


def main():
    # Start Mlflow server 
    # infra_server = Server()
    # infra_server.start()

    # time.sleep(15)

    # MLFLOW_TRACKING_URI = "http://localhost:5001/"

    # try:
    #     requests.get(MLFLOW_TRACKING_URI)
    # except:
    #     raise requests.exceptions.ConnectionError("Unable to reach MLOps.")
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    
    MODEL = ResNetV3_WInit()
    VAL_SIZE = 0.05
    BATCH_SIZE = 256
    NUM_EPOCHS = 2
    EXPERIMENT_NAME = "resnet-style-cnn"
    
    DATAMODULE = CIFAR10DataModule(
        val_size=VAL_SIZE, 
        batch_size=BATCH_SIZE, 
        transform=None
    )

    run_pipeline(
        model=MODEL, 
        datamodule=DATAMODULE, 
        num_epochs=NUM_EPOCHS,
        experiment_name=EXPERIMENT_NAME, 
    )


if __name__ == "__main__":
    main()
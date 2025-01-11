import os
from ml.models import VGGStyleV3
from ml.pipelines import run_pipeline


if __name__ == "__main__":
    MODEL = VGGStyleV3()
    BATCH_SIZE = 64
    NUM_EPOCHS = 180
    NUM_WORKERS = os.cpu_count()
    VAL_SIZE = 0.05
    EXPERIMENT_NAME = "vgg-style-cnn"
    RANDOM_STATE = 0

    run_pipeline(
        MODEL, 
        BATCH_SIZE, 
        NUM_EPOCHS, 
        NUM_WORKERS, 
        VAL_SIZE, 
        EXPERIMENT_NAME, 
        RANDOM_STATE
    )
    
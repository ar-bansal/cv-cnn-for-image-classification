from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from ml.models import VGGStyleV1
from ml.pipelines import run_pipeline


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_EPOCHS = 5

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = CIFAR10(root="./data/train", train=True, transform=transform)
    test_dataset = CIFAR10(root="./data/test", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_labels = CIFAR10.classes
   
    run_pipeline(
        VGGStyleV1(), 
        NUM_EPOCHS, 
        train_loader, 
        test_loader, 
        class_labels, 
        experiment_name="vgg-style-cnn"
    )


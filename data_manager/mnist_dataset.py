import glob
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_dataset(im_path, is_train=True):
    """
    Get dataset from torchvision
    :param im_path: store dataset, downloaded dataset will be stored
    :param split: mode
    """
    dataset = torchvision.datasets.MNIST(
        root=im_path,
        train=is_train,
        download=True,
        transform=transform
    )

    print("Total images:", len(dataset))
    return dataset

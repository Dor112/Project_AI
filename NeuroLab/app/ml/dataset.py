from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

FASHION_MNIST_CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


def get_transforms():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def prepare_fashion_mnist(root, batch_size=64, val_fraction=0.1, download=True,
                          num_workers=0, train_subset_size=None, test_subset_size=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    transform = get_transforms()
    train_full = datasets.FashionMNIST(root=str(root), train=True, download=download, transform=transform)
    test_full = datasets.FashionMNIST(root=str(root), train=False, download=download, transform=transform)

    if train_subset_size and train_subset_size < len(train_full):
        train_full = Subset(train_full, list(range(train_subset_size)))
    if test_subset_size and test_subset_size < len(test_full):
        test_full = Subset(test_full, list(range(test_subset_size)))

    n_total = len(train_full)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_full
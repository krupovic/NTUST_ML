import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_mnist_loaders(batch_size: int = 128, val_split: float = 0.1, max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # optionally limit samples from the training set for quick tests
    if max_samples is not None:
        train_full = Subset(train_full, list(range(min(len(train_full), max_samples))))
        test = Subset(test, list(range(min(len(test), max_samples))))

    # split train_full into train and val
    if val_split is not None and val_split > 0:
        n = len(train_full)
        val_n = int(n * val_split)
        train_n = n - val_n
        train_ds, val_ds = torch.utils.data.random_split(train_full, [train_n, val_n])
    else:
        train_ds = train_full
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

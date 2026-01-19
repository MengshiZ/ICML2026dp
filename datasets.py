"""Dataset loading for the centralized experiments (PyTorch-only).

This repo originally mixed TensorFlow TFDS + PyTorch. For simplicity and
portability, this module uses only torchvision datasets.

We match the original code's input scaling: uint8 images were mapped to
[-1, 1] via x/127.5 - 1.0.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _default_transform(channels: int) -> transforms.Compose:
    # ToTensor() maps uint8 [0,255] -> float32 [0,1].
    # Normalize((0.5,), (0.5,)) maps [0,1] -> [-1,1].
    if channels == 1:
        mean, std = (0.5,), (0.5,)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_data(name: str, root: str = "./data") -> Tuple[Dataset, Dataset, int, int]:
    """Load a dataset.

    Args:
        name: One of {'mnist', 'cifar10', 'emnist_merge'}.
        root: Where to download/store the data.

    Returns:
        trainset, testset, ntrain, nclass
    """
    name = name.lower()
    if name == "mnist":
        tfm = _default_transform(channels=1)
        trainset = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
        testset = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
        nclass = 10
    elif name == "cifar10":
        tfm = _default_transform(channels=3)
        trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
        testset = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
        nclass = 10
    elif name in {"emnist_merge", "emnist_bymerge", "emnist"}:
        # Paper: EMNIST (ByMerge split).
        tfm = _default_transform(channels=1)
        trainset = datasets.EMNIST(root=root, split="bymerge", train=True, download=True, transform=tfm)
        testset = datasets.EMNIST(root=root, split="bymerge", train=False, download=True, transform=tfm)
        # ByMerge has 47 classes.
        nclass = 47
    else:
        raise ValueError(f"Unknown dataset: {name}")

    ntrain = len(trainset)
    return trainset, testset, ntrain, nclass

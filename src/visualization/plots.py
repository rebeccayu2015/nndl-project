"""
plots.py
---------

General-purpose visualization utilities used across the project
"""

import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import make_grid
import torch
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
import numpy as np

def plot_superclass_distribution(train_csv_path: str, save_path: str):
    df = pd.read_csv(train_csv_path)
    counts = df['superclass_index'].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Superclass index")
    ax.set_ylabel("# Samples")
    ax.set_title("Superclass distribution (train)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1, 3, 1, 1)

    if img.dim() == 3:
        img = img.unsqueeze(0)  # (1, C, H, W)

    img = img * std + mean
    img = img.clamp(0.0, 1.0)
    return img.squeeze(0) if img.size(0) == 1 else img


def plot_sample_images(dataloader, save_path: str, n_images: int = 16):
    images, y_super, y_sub, _ = next(iter(dataloader))
    images = images[:n_images]

    # Make grid, then unnormalize the whole grid tensor
    grid = make_grid(images, nrow=4, padding=2)  # (C, H, W)
    grid_unnorm = _unnormalize(grid)            # still (C, H, W)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_unnorm.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    ax.set_title("Sample training images")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_curves(
    train_losses,
    val_losses,
    save_path: str,
    title: str = "Training and validation loss",
):
    # Convert to numpy arrays for safety
    train_losses = np.asarray(train_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)

    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_accuracy_curves(
    train_acc,
    val_acc,
    save_path: str,
    title: str = "Training and validation accuracy",
):
    train_acc = np.asarray(train_acc, dtype=float)
    val_acc = np.asarray(val_acc, dtype=float)

    epochs = np.arange(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_acc, label="Train accuracy")
    ax.plot(epochs, val_acc, label="Val accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
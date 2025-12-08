"""
Visual sanity-check for dataset + transforms.

1. Loads your training dataset with augmentation.
2. Fetches a batch of images.
3. Displays them in a grid so you can visually inspect:
   - cropping
   - rotations
   - flips
   - color jitter
   - normalization correctness after un-normalizing
4. Confirms labels + metadata look correct.
"""

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from src.data.dataset import BirdDogReptileDataset
from src.data.transforms import get_train_transform, IMAGENET_MEAN, IMAGENET_STD
from torch.utils.data import DataLoader

# Helper to un-normalize image tensors for visual display
def unnormalize(img_tensor):
    """
    img_tensor: (3, H, W)
    Returns numpy array in range [0,1] for visualization.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def visualize_batch(images, super_labels, sub_labels, filenames):
    # Unnormalize each image
    imgs = [unnormalize(images[i]) for i in range(images.size(0))]

    # Stack into grid
    grid = make_grid(images, nrow=4)
    grid = unnormalize(grid)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.title("Augmented Training Images (Unnormalized for Display)")
    plt.axis("off")
    plt.show()

    # Print labels + filenames for debugging
    print("Superclass labels:", super_labels[:8])
    print("Subclass labels:", sub_labels[:8])
    print("Filenames:", filenames[:8])


if __name__ == "__main__":
    # Load dataset with train transform (augmentation)
    train_ds = BirdDogReptileDataset(
        csv_path="data/splits/train_split.csv",
        images_root="data/raw/train_images",
        transform=get_train_transform(),
        superclass_mapping_path="data/meta/superclass_mapping.csv",
        subclass_mapping_path="data/meta/subclass_mapping.csv",
    )

    # DataLoader
    dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Fetch a batch
    images, y_super, y_sub, meta = next(iter(dl))

    print("Batch images shape:", images.shape)
    print("First filename:", meta["image"][0])

    visualize_batch(images, y_super, y_sub, meta["image"])

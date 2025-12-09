"""
generate_sample_image_grid.py
------------------------------

Uses plot_sample_images() from plots.py to save a grid of dataset images
"""

import os
from torch.utils.data import DataLoader

from src.data.dataset import BirdDogReptileDataset
from src.data.transforms import get_eval_transform, get_train_transform
from src.visualization.plots import plot_sample_images

def main():
    os.makedirs("experiments/vis", exist_ok=True)

    dataset = BirdDogReptileDataset(
        csv_path="data/splits/train_split.csv",
        images_root="data/raw/train_images",
        transform=get_train_transform(),
        superclass_mapping_path="data/meta/superclass_mapping.csv",
        subclass_mapping_path="data/meta/subclass_mapping.csv",
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    save_path = "experiments/vis/sample_images_from_plots.png"

    plot_sample_images(loader, save_path=save_path, n_images=16)

    print(f"Saved sample images plot → {save_path}")

if __name__ == "__main__":
    main()

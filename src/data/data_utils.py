"""
data_utils.py
--------------

Utility functions for preparing and managing dataset metadata used by
the custom Dataset class

1. Creates train/val/calibration splits from train_data.csv, saving them into the resulting CSVs (train_split.csv, val_split.csv,
     calibration_split.csv)
    - First, split into train and val (stratified by superclass_index).
    - Then, carve calibration from the training portion.

2. Loads and validates label mappings in superclass_mapping.csv and subclass_mapping.csv.

3. General metadata utilities
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100

def create_splits(
    train_csv_path: str,
    output_dir: str,
    val_size: float = 0.15,
    calib_size_from_train: float = 0.10,
    random_state: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(train_csv_path)

    assert 'superclass_index' in df.columns

    # Train/Val split
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df['superclass_index']
    )

    # Calibration from train
    calib_df, new_train_df = train_test_split(
        train_df,
        test_size=(1.0 - calib_size_from_train),
        random_state=random_state,
        stratify=train_df['superclass_index']
    )

    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    calib_path = os.path.join(output_dir, "calibration_split.csv")

    new_train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    calib_df.to_csv(calib_path, index=False)

    print(f"Saved train split to {train_path} ({len(new_train_df)} samples)")
    print(f"Saved val split to {val_path} ({len(val_df)} samples)")
    print(f"Saved calibration split to {calib_path} ({len(calib_df)} samples)")

ANIMAL_FINE_LABELS = {
    "beaver", "dolphin", "otter", "seal", "whale",

    "rabbit", "hamster", "mouse", "shrew", "squirrel",

    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",

    "bear", "lion", "tiger", "leopard",

    "butterfly", "caterpillar", "cockroach",

    "baby_chicken", "bird", "canary",

    "crocodile", "dinosaur", "lizard", "snake", "turtle",
}

EXCLUDE_BIRD_REPTILE = {
    "baby_chicken", "bird", "canary",
    "crocodile", "dinosaur", "lizard", "snake", "turtle",
}

def export_cifar100(
    out_dir="data/raw/farood_images",
    train=False,
    max_items=5000,
    include_labels=ANIMAL_FINE_LABELS,
    exclude_labels=EXCLUDE_BIRD_REPTILE,
    clear_dir=True,
):
    os.makedirs(out_dir, exist_ok=True)

    if clear_dir:
        for f in glob.glob(os.path.join(out_dir, "*.jpg")):
            os.remove(f)

    ds = CIFAR100(root="./data", train=train, download=True)
    classes = ds.classes

    saved = 0
    included_seen = set()
    excluded_seen = set()

    for i in range(len(ds)):
        img, y = ds[i]
        label_name = classes[y]

        if label_name not in include_labels:
            continue

        if label_name in exclude_labels:
            excluded_seen.add(label_name)
            continue

        included_seen.add(label_name)

        img.save(
            os.path.join(out_dir, f"{saved}.jpg"),
            format="JPEG",
            quality=95,
            optimize=True,
        )
        saved += 1
        if max_items is not None and saved >= max_items:
            break

    print(f"Saved {saved} animal-only (filtered) CIFAR-100 images to {out_dir}")
    print("Included labels used:", sorted(included_seen))
    print("Excluded labels encountered:", sorted(excluded_seen))


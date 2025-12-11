"""
plot_training_curves.py
-----------------------

Utility script to generate training/validation loss and accuracy curves
from the recorded <backbone>_metrics.csv files produced during training.

Usage:
    python -m src.visualization.plot_training_curves --backbone resnet50
"""

import os
import argparse
import pandas as pd

from src.visualization.plots import (
    plot_loss_curves,
    plot_accuracy_curves,
)


def load_metrics(csv_path: str):
    """Load training metrics CSV into a Pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find metrics file: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def plot_all_curves(df, backbone: str, out_dir: str):
    """Generate all plots for loss, superclass accuracy, and subclass accuracy."""
    os.makedirs(out_dir, exist_ok=True)

    # ---- Extract metrics ----
    train_loss = df["train_loss"].tolist()
    val_loss   = df["val_loss"].tolist()

    train_super = df["train_super_acc"].tolist()
    val_super   = df["val_super_acc"].tolist()

    train_sub = df["train_sub_acc"].tolist()
    val_sub   = df["val_sub_acc"].tolist()

    # ---- Plot loss curves ----
    loss_path = os.path.join(out_dir, f"{backbone}_loss_curves.png")
    plot_loss_curves(
        train_loss,
        val_loss,
        save_path=loss_path,
        title=f"{backbone} - Training vs Validation Loss",
    )

    # ---- Superclass accuracy curves ----
    super_path = os.path.join(out_dir, f"{backbone}_super_accuracy.png")
    plot_accuracy_curves(
        train_super,
        val_super,
        save_path=super_path,
        title=f"{backbone} - Superclass Accuracy",
    )

    # ---- Subclass accuracy curves ----
    sub_path = os.path.join(out_dir, f"{backbone}_sub_accuracy.png")
    plot_accuracy_curves(
        train_sub,
        val_sub,
        save_path=sub_path,
        title=f"{backbone} - Subclass Accuracy",
    )

    print(f"\nSaved training curves for {backbone}:")
    print(" •", loss_path)
    print(" •", super_path)
    print(" •", sub_path)


def main(cfg):
    metrics_csv = os.path.join("checkpoints", f"{cfg.backbone}_metrics.csv")

    print("Loading metrics from:", metrics_csv)
    df = load_metrics(metrics_csv)

    out_dir = os.path.join("figures", "training_curves")
    plot_all_curves(df, cfg.backbone, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "densenet121", "efficientnet_b0"],
        help="Which model's metrics CSV to plot."
    )

    cfg = parser.parse_args()
    main(cfg)

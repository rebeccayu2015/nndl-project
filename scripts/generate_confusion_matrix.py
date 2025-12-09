"""
generate_confusion_matrix.py
-----------------------------

Creates and saves a confusion matrix plot using utilities from cm.py
"""

import os
import numpy as np
from src.visualization.cm import plot_confusion_matrix

def main():
    os.makedirs("experiments/vis", exist_ok=True)

    #  Example data
    y_true = np.array([0, 1, 2, 1, 0, 2, 2, 1])
    y_pred = np.array([0, 2, 2, 1, 0, 1, 2, 1])

    # y_true = np.load("experiments/baseline/val_labels.npy")
    # y_pred = np.load("experiments/baseline/val_preds.npy")

    class_names = ["bird", "dog", "reptile"]

    save_path = "experiments/vis/confusion_matrix_example.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        title="Example Confusion Matrix",
        save_path=save_path
    )

    print(f"Saved confusion matrix → {save_path}")

if __name__ == "__main__":
    main()

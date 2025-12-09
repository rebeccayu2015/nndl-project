"""
generate_class_distribution_plot.py
------------------------------------

Creates a class distribution plot using plot_superclass_distribution()
from plots.py
"""

import os
from src.visualization.plots import plot_superclass_distribution

def main():
    os.makedirs("experiments/vis", exist_ok=True)

    csv_path = "data/meta/train_data.csv"
    save_path = "experiments/vis/superclass_distribution.png"

    plot_superclass_distribution(csv_path, save_path)

    print(f"Saved superclass distribution plot → {save_path}")

if __name__ == "__main__":
    main()

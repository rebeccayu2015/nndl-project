import json
import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: Path):
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_metrics(csv_path: Path, model_name: str, metrics: Dict[str, float]):
    """
    Append a metrics row to csv_path with a fixed header layout.
    """
    ensure_dir(csv_path.parent)

    header = [
        "model",
        "cross_entropy_super",
        "cross_entropy_sub",
        "cross_entropy_total",
        "super_accuracy",
        "super_accuracy_seen",
        "super_accuracy_unseen",
        "sub_accuracy",
        "sub_accuracy_seen",
        "sub_accuracy_unseen",
    ]

    row = [
        model_name,
        metrics.get("cross_entropy_super"),
        metrics.get("cross_entropy_sub"),
        metrics.get("cross_entropy_total"),
        metrics.get("super_accuracy"),
        metrics.get("super_accuracy_seen"),
        metrics.get("super_accuracy_unseen"),
        metrics.get("sub_accuracy"),
        metrics.get("sub_accuracy_seen"),
        metrics.get("sub_accuracy_unseen"),
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

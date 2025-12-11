# src/inference/eval_dual_head.py

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.models.dual_head import DualHeadNet
from src.data.dataset import BirdDogReptileDataset
from src.visualization.cm import plot_confusion_matrix


def evaluate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Same transforms as training
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # Dataset + DataLoader (usually val or test split)
    ds = BirdDogReptileDataset(cfg.csv_path, cfg.images_root, transform=transform)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # Model
    model = DualHeadNet(
        backbone_name=cfg.backbone,
        num_super=cfg.num_super,
        num_sub=cfg.num_sub,
        pretrained=False,          # weights come from checkpoint
        freeze_backbone=False,
    ).to(device)

    # Load checkpoint
    ckpt_path = cfg.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join("checkpoints", f"best_{cfg.backbone}.pth")
    print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_super_true = []
    all_super_pred = []
    all_sub_true = []
    all_sub_pred = []

    with torch.no_grad():
        for imgs, y_super, y_sub, _ in dl:
            imgs = imgs.to(device)
            y_super = y_super.to(device)
            y_sub = y_sub.to(device)

            logits_super, logits_sub = model(imgs)

            pred_super = torch.argmax(logits_super, dim=1)
            pred_sub   = torch.argmax(logits_sub, dim=1)

            all_super_true.append(y_super.cpu())
            all_super_pred.append(pred_super.cpu())
            all_sub_true.append(y_sub.cpu())
            all_sub_pred.append(pred_sub.cpu())

    all_super_true = torch.cat(all_super_true).numpy()
    all_super_pred = torch.cat(all_super_pred).numpy()
    all_sub_true   = torch.cat(all_sub_true).numpy()
    all_sub_pred   = torch.cat(all_sub_pred).numpy()

    # Simple accuracies
    super_acc = (all_super_true == all_super_pred).mean()
    sub_acc   = (all_sub_true == all_sub_pred).mean()

    print(f"\n[{cfg.backbone}] Evaluation on {cfg.csv_path}")
    print(f"Superclass accuracy: {super_acc:.4f}")
    print(f"Subclass accuracy:   {sub_acc:.4f}")

    # Confusion matrices
    os.makedirs(cfg.out_dir, exist_ok=True)

    super_classes = [f"super_{i}" for i in range(cfg.num_super)]
    sub_classes   = [f"sub_{i}" for i in range(cfg.num_sub)]

    super_cm_path = os.path.join(cfg.out_dir, f"{cfg.backbone}_super_cm.png")
    sub_cm_path   = os.path.join(cfg.out_dir, f"{cfg.backbone}_sub_cm.png")

    print(f"Saving superclass confusion matrix to: {super_cm_path}")
    plot_confusion_matrix(
        all_super_true,
        all_super_pred,
        class_names=super_classes,
        title=f"{cfg.backbone} - Superclass CM",
        save_path=super_cm_path,
    )

    print(f"Saving subclass confusion matrix to: {sub_cm_path}")
    plot_confusion_matrix(
        all_sub_true,
        all_sub_pred,
        class_names=sub_classes,
        title=f"{cfg.backbone} - Subclass CM",
        save_path=sub_cm_path,
    )


if __name__ == "__main__":
    import torch  # needed here for cat / no_grad in top scope

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV with image, superclass_index, subclass_index")
    parser.add_argument("--images_root", type=str, required=True,
                        help="Root folder where images live")

    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "densenet121", "efficientnet_b0"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Defaults to checkpoints/best_<backbone>.pth")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_super", type=int, default=4)
    parser.add_argument("--num_sub", type=int, default=88)

    parser.add_argument("--out_dir", type=str, default="figures/confusion_matrices")

    cfg = parser.parse_args()
    evaluate(cfg)

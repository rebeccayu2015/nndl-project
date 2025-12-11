import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.models.dual_head import DualHeadNet
from src.data.dataset import BirdDogReptileDataset



# -------------------------------------------------------------
# Utility: accuracy function
# -------------------------------------------------------------
def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


# -------------------------------------------------------------
# Training step (one epoch)
# -------------------------------------------------------------
def train_one_epoch(model, dl, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_super_acc = 0
    total_sub_acc = 0
    n = 0

    for imgs, y_super, y_sub, _ in dl:
        imgs = imgs.to(device)
        y_super = y_super.to(device)
        y_sub = y_sub.to(device)

        logits_super, logits_sub = model(imgs)

        loss_super = criterion(logits_super, y_super)
        loss_sub   = criterion(logits_sub, y_sub)
        loss = loss_super + loss_sub

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        n += batch_size
        total_loss += loss.item() * batch_size
        total_super_acc += accuracy(logits_super, y_super) * batch_size
        total_sub_acc   += accuracy(logits_sub, y_sub) * batch_size

    return (
        total_loss / n,
        total_super_acc / n,
        total_sub_acc / n,
    )


# -------------------------------------------------------------
# Evaluation step (one epoch)
# -------------------------------------------------------------
@torch.no_grad()
def eval_one_epoch(model, dl, criterion, device):
    model.eval()
    total_loss = 0
    total_super_acc = 0
    total_sub_acc = 0
    n = 0

    for imgs, y_super, y_sub, _ in dl:
        imgs = imgs.to(device)
        y_super = y_super.to(device)
        y_sub = y_sub.to(device)

        logits_super, logits_sub = model(imgs)

        loss_super = criterion(logits_super, y_super)
        loss_sub   = criterion(logits_sub, y_sub)
        loss = loss_super + loss_sub

        batch_size = imgs.size(0)
        n += batch_size
        total_loss += loss.item() * batch_size
        total_super_acc += accuracy(logits_super, y_super) * batch_size
        total_sub_acc   += accuracy(logits_sub, y_sub) * batch_size

    return (
        total_loss / n,
        total_super_acc / n,
        total_sub_acc / n,
    )


# -------------------------------------------------------------
# Main training function
# -------------------------------------------------------------
def train_transfer(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global global_epoch
    global_epoch = 0
    transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

    # Create save directory
    os.makedirs(cfg.save_dir, exist_ok=True)

    # save metrics for comparison
    import csv

    metrics_path = os.path.join(cfg.save_dir, f"{cfg.backbone}_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "phase",
            "train_loss",
            "train_super_acc",
            "train_sub_acc",
            "val_loss",
            "val_super_acc",
            "val_sub_acc"
        ])

    # -----------------------------------------
    # Dataset + loaders
    # -----------------------------------------
    train_ds = BirdDogReptileDataset(cfg.train_csv, cfg.images_root, transform=transform)
    val_ds   = BirdDogReptileDataset(cfg.val_csv, cfg.images_root, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # -----------------------------------------
    # Model
    # -----------------------------------------
    model = DualHeadNet(
        backbone_name=cfg.backbone,
        num_super=cfg.num_super,
        num_sub=cfg.num_sub,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------
    # 1. Frozen backbone phase
    # -------------------------------------------------------
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n=== Frozen backbone training ===")
    for epoch in range(cfg.epochs_frozen):

        tr = train_one_epoch(model, train_dl, optimizer, criterion, device)
        va = eval_one_epoch(model, val_dl, criterion, device)

        print(f"[Frozen {epoch:02d}] train loss={tr[0]:.4f}, super_acc={tr[1]:.4f}, sub_acc={tr[2]:.4f} | "
              f"val loss={va[0]:.4f}, super_acc={va[1]:.4f}, sub_acc={va[2]:.4f}")
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
            global_epoch,
            "frozen",
            tr[0], tr[1], tr[2],
            va[0], va[1], va[2],
            ])
            global_epoch += 1

        # Early stopping logic
        if va[0] < best_val_loss:
            best_val_loss = va[0]
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(cfg.save_dir, f"best_{cfg.backbone}.pth"))
            print("  -> New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("  -> Early stopping triggered during frozen phase.")
                break

    # -------------------------------------------------------
    # 2. Fine-tuning phase (unfreeze backbone)
    # -------------------------------------------------------
    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs_tune)

    print("\n=== Fine-tuning backbone ===")
    # Reset early stopping for fine-tuning phase
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(cfg.epochs_tune):

        tr = train_one_epoch(model, train_dl, optimizer, criterion, device)
        va = eval_one_epoch(model, val_dl, criterion, device)

        print(f"[Tune {epoch:02d}] train loss={tr[0]:.4f}, super_acc={tr[1]:.4f}, sub_acc={tr[2]:.4f} | "
              f"val loss={va[0]:.4f}, super_acc={va[1]:.4f}, sub_acc={va[2]:.4f}")
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
            global_epoch,
            "tune",
            tr[0], tr[1], tr[2],
            va[0], va[1], va[2],
            ])
            global_epoch += 1

        scheduler.step()

        # Early stopping
        if va[0] < best_val_loss:
            best_val_loss = va[0]
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(cfg.save_dir, f"best_{cfg.backbone}.pth"))
            print("  -> New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("  -> Early stopping triggered during fine-tuning.")
                break

    print(f"\nTraining complete. Best model saved as best_{cfg.backbone}.pth")


# -------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=True)

    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "densenet121", "efficientnet_b0"])

    parser.add_argument("--num_super", type=int, default=4)
    parser.add_argument("--num_sub", type=int, default=88)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--epochs_frozen", type=int, default=5)
    parser.add_argument("--epochs_tune", type=int, default=10)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    cfg = parser.parse_args()
    train_transfer(cfg)
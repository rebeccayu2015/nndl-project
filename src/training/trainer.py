import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, Optional
from tqdm import tqdm

from src.models.baseline_cnn import BaselineCNN
from src.models.dual_head import DualHeadNet
from src.data.dataset import BirdDogReptileDataset, NearOODDataset, FarOODDataset, TestDataset
from src.data.transforms import get_train_transform, get_eval_transform, get_clip_transform, get_near_ood_transform
from src.training.configs import TrainingConfig
from src.utils.const import NOVEL_SUPER_IDX, NOVEL_SUB_IDX

def build_dataloaders(cfg: TrainingConfig):
    if cfg.mode == "clip_b32":
        train_tf = get_clip_transform()
        val_tf   = get_clip_transform()
    else:
        train_tf = get_train_transform_cnn()
        val_tf   = get_eval_transform_cnn()

    train_ds = BirdDogReptileDataset(
        csv_path=cfg.train_csv,
        images_root=cfg.images_root,
        transform=train_tf,
        superclass_mapping_path=cfg.superclass_map,
        subclass_mapping_path=cfg.subclass_map,
    )

    val_ds = BirdDogReptileDataset(
        csv_path=cfg.val_csv,
        images_root=cfg.images_root,
        transform=val_tf,
        superclass_mapping_path=cfg.superclass_map,
        subclass_mapping_path=cfg.subclass_map,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # OSR Calibration
    calib_ds = BirdDogReptileDataset(
            csv_path=cfg.calib_csv,
            images_root=cfg.images_root,
            transform=val_tf,
            superclass_mapping_path=cfg.superclass_map,
            subclass_mapping_path=cfg.subclass_map,
        )
    
    calib_loader = DataLoader(
        calib_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    near_ood_ds = NearOODDataset(
        base_dataset=BirdDogReptileDataset(
            csv_path=cfg.calib_csv,
            images_root=cfg.images_root,
            transform=None,
            superclass_mapping_path=cfg.superclass_map,
            subclass_mapping_path=cfg.subclass_map,
        ),
        transform=get_near_ood_transform(),
    )
    
    near_ood_loader = DataLoader(
        near_ood_ds, 
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    far_ood_ds = FarOODDataset(
            images_root=cfg.far_ood_root,
            transform=val_tf,
        )
    
    far_ood_loader = DataLoader(
        far_ood_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader, calib_loader, near_ood_loader, far_ood_loader


def build_model(cfg: TrainingConfig):
    return DualHeadNet(
        backbone_name=cfg.mode,
        num_super=cfg.num_super,
        num_sub=cfg.num_sub,
        pretrained=True,
        freeze_backbone=True,
    )
    

def build_optimizer(name: str, params, lr: float):
    name = (name or "").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(name: Optional[str], optimizer, epochs: Optional[int]):
    if name is None or name == "none" or epochs is None:
        return None
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {name}")


class EarlyStopper:
    def __init__(self, patience: Optional[int]):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def update(self, value: float) -> bool:
        if self.patience is None:
            return False
        if value < self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        test_loader=None,
        device: str | torch.device = "cuda",
        lambda_sub: float = 1.0,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        patience: Optional[int] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lambda_sub = lambda_sub
        self.device = torch.device(device)
        self.patience = patience

        self.model.to(self.device)

    def _step(self, batch, train: bool) -> Dict[str, float]:
        images, y_super, y_sub, _ = batch
        images = images.to(self.device)
        y_super = y_super.to(self.device)
        y_sub = y_sub.to(self.device)

        if train:
            self.optimizer.zero_grad()

        logits_super, logits_sub = self.model(images)

        loss_super = self.criterion(logits_super, y_super)
        loss_sub = self.criterion(logits_sub, y_sub)
        loss = loss_super + self.lambda_sub * loss_sub

        if train:
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            super_pred = logits_super.argmax(dim=1)
            sub_pred = logits_sub.argmax(dim=1)

            super_acc = (super_pred == y_super).float().mean().item()
            sub_acc = (sub_pred == y_sub).float().mean().item()

        return {
            "loss": loss.item(),
            "super_accuracy": super_acc,
            "sub_accuracy": sub_acc,
            "count": images.size(0),
        }

    def _run_epoch(self, loader, train: bool) -> Dict[str, float]:
        totals = {
            "loss": 0.0,
            "super_accuracy": 0.0,
            "sub_accuracy": 0.0,
            "count": 0,
        }

        phase = "Train" if train else "Val"
        pbar = tqdm(loader, desc=phase, leave=False)

        for batch in pbar:
            stats = self._step(batch, train=train)

            bs = stats["count"]
            totals["count"] += bs
            totals["loss"] += stats["loss"] * bs
            totals["super_accuracy"] += stats["super_accuracy"] * bs
            totals["sub_accuracy"] += stats["sub_accuracy"] * bs

            # running averages
            avg_loss = totals["loss"] / totals["count"]
            avg_super = totals["super_accuracy"] / totals["count"]
            avg_sub = totals["sub_accuracy"] / totals["count"]

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                super_acc=f"{avg_super:.3f}",
                sub_acc=f"{avg_sub:.3f}",
            )

        if train and self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": totals["loss"] / totals["count"],
            "super_accuracy": totals["super_accuracy"] / totals["count"],
            "sub_accuracy": totals["sub_accuracy"] / totals["count"],
        }
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        return self._run_epoch(self.train_loader, train=True)

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        return self._run_epoch(self.val_loader, train=False)

    @torch.no_grad()
    def test_epoch(self) -> Dict[str, float]:
        if self.test_loader is None:
            raise ValueError("No test_loader provided")
        self.model.eval()
        return self._run_epoch(self.test_loader, train=False)

    def fit(
        self,
        epochs: int,
        phase: str = "Epoch",
        early_stopper: Optional[EarlyStopper] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        best_val_loss = float("inf")
    
        for epoch in range(epochs):
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch()
    
            if verbose:
                print(
                    f"[{phase} {epoch:02d}] "
                    f"train_loss={train_stats['loss']:.4f}, "
                    f"val_loss={val_stats['loss']:.4f}, "
                    f"val_super_acc={val_stats['super_accuracy']:.4f}, "
                    f"val_sub_acc={val_stats['sub_accuracy']:.4f}"
                )
    
            if early_stopper is not None:
                if early_stopper.update(val_stats["loss"]):
                    print(f"Early stopping during {phase_name.lower()} training.")
                    break
    
            if save_path is not None and val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                torch.save(self.model.state_dict(), save_path)

    @torch.no_grad()
    def _collect_confidence(self, loader):
        super_scores, sub_scores = [], []
        for images, _, _, _ in loader:
            images = images.to(self.device)
            logits_super, logits_sub = self.model(images)
            
            super_conf = F.softmax(logits_super, dim=1).max(dim=1).values
            sub_conf   = F.softmax(logits_sub, dim=1).max(dim=1).values

            super_scores.extend(super_conf.cpu())
            sub_scores.extend(sub_conf.cpu())
        return torch.stack(super_scores), torch.stack(sub_scores)
        
    @torch.no_grad()
    def calibrate(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        id_quantile: float = 0.05,
        ood_quantile: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calibrate open-set thresholds using ID + OOD data.
        """
        self.model.eval()
        
        id_super, id_sub   = self._collect_confidence(id_loader)
        ood_super, ood_sub = self._collect_confidence(ood_loader)

        tau_super = float(
            (torch.quantile(id_super, id_quantile)
           + torch.quantile(ood_super, ood_quantile)) / 2
        )

        tau_sub = float(
            (torch.quantile(id_sub, id_quantile)
           + torch.quantile(ood_sub, ood_quantile)) / 2
        )

        self.tau_super = tau_super
        self.tau_sub = tau_sub

        return {
            "tau_super": tau_super,
            "tau_sub": tau_sub,
        }

    @torch.no_grad()
    def evaluate_open_set(self, dataloader: DataLoader) -> Dict[str, float]:
        assert hasattr(self, "tau_super"), "Call calibrate() first"
        assert hasattr(self, "tau_sub"), "Call calibrate() first"

        self.model.eval()

        ce_loss = torch.nn.CrossEntropyLoss(reduction="sum")

        ce_super = 0.0
        ce_sub = 0.0

        total = 0
        seen_super = unseen_super = 0
        seen_sub = unseen_sub = 0

        super_correct = seen_super_correct = unseen_super_correct = 0
        sub_correct = seen_sub_correct = unseen_sub_correct = 0

        for images, y_super, y_sub, _ in dataloader:
            images = images.to(self.device)
            y_super = y_super.to(self.device)
            y_sub = y_sub.to(self.device)

            logits_super, logits_sub = self.model(images)

            ce_super += ce_loss(logits_super, y_super).item()
            ce_sub += ce_loss(logits_sub, y_sub).item()

            pred_super = logits_super.argmax(dim=1)
            pred_sub = logits_sub.argmax(dim=1)

            # OSR 
            probs_super = torch.softmax(logits_super, dim=1)
            probs_sub   = torch.softmax(logits_sub, dim=1)
    
            conf_super, pred_super = probs_super.max(dim=1)
            conf_sub, pred_sub     = probs_sub.max(dim=1)
    
            pred_super[conf_super < self.tau_super] = NOVEL_SUPER_IDX
            pred_sub[conf_sub < self.tau_sub] = NOVEL_SUB_IDX
    
            pred_sub[pred_super == NOVEL_SUPER_IDX] = NOVEL_SUB_IDX

            for i in range(len(y_super)):
                total += 1

                is_super_unseen = y_super[i] == NOVEL_SUPER_IDX
                is_sub_unseen = y_sub[i] == NOVEL_SUB_IDX

                super_correct += (pred_super[i] == y_super[i]).item()
                if is_super_unseen:
                    unseen_super += 1
                    unseen_super_correct += (pred_super[i] == y_super[i]).item()
                else:
                    seen_super += 1
                    seen_super_correct += (pred_super[i] == y_super[i]).item()

                sub_correct += (pred_sub[i] == y_sub[i]).item()
                if is_sub_unseen:
                    unseen_sub += 1
                    unseen_sub_correct += (pred_sub[i] == y_sub[i]).item()
                else:
                    seen_sub += 1
                    seen_sub_correct += (pred_sub[i] == y_sub[i]).item()

        return {
            "ce_super": ce_super / total,
            "ce_sub": ce_sub / total,

            "super_overall": 100 * super_correct / total,
            "super_seen": 100 * seen_super_correct / max(1, seen_super),
            "super_unseen": 100 * unseen_super_correct / max(1, unseen_super),

            "sub_overall": 100 * sub_correct / total,
            "sub_seen": 100 * seen_sub_correct / max(1, seen_sub),
            "sub_unseen": 100 * unseen_sub_correct / max(1, unseen_sub),
        }

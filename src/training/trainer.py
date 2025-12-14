import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, Optional, Union
from tqdm import tqdm
import json
import os

from src.models.dual_head import DualHeadNet
from src.data.dataset import (
    BirdDogReptileDataset,
    NearOODDataset,
    FarOODDataset,
    TestDataset,
)
from src.data.transforms import (
    get_train_transform,
    get_eval_transform,
    get_clip_transform,
    get_near_ood_transform,
)
from src.training.configs import TrainingConfig
from src.utils.const import NOVEL_SUPER_IDX, NOVEL_SUB_IDX


# =============================================================
# DATALOADERS
# =============================================================
def build_dataloaders(cfg: TrainingConfig):
    if cfg.mode == "clip_b32":
        train_tf = get_clip_transform()
        val_tf = get_clip_transform()
    else:
        train_tf = get_train_transform()
        val_tf = get_eval_transform()

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

    # --- OSR calibration loaders ---
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


# =============================================================
# MODEL / OPTIMIZER / SCHEDULER
# =============================================================
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
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    raise ValueError(f"Unsupported scheduler: {name}")


# =============================================================
# EARLY STOPPER
# =============================================================
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


# =============================================================
# TRAINER
# =============================================================
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        test_loader=None,
        device: Union[str, torch.device] = "cuda",
        lambda_sub: float = 1.0,
        scheduler: Optional[
            torch.optim.lr_scheduler._LRScheduler
        ] = None,
        patience: Optional[int] = None,
        run_path: Optional[str] = None,
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
        self.run_path = run_path

        self.model.to(self.device)

    # ---------------------------------------------------------
    # TRAINING STEP
    # ---------------------------------------------------------
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
            "loss_super": loss_super.item(),
            "loss_sub": loss_sub.item(),
            "super_accuracy": super_acc,
            "sub_accuracy": sub_acc,
            "count": images.size(0),
        }

    # ---------------------------------------------------------
    # PROTOTYPES
    # ---------------------------------------------------------
    @torch.no_grad()
    def compute_prototypes(
        self,
        loader: DataLoader,
        level: str = "sub",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        feats_by_class = {}

        for images, y_super, y_sub, _ in loader:
            images = images.to(self.device)
            feats = self.model.extract_features(images)
            labels = y_sub if level == "sub" else y_super
            labels = labels.to(self.device)

            for f, y in zip(feats, labels):
                y = int(y.item())
                feats_by_class.setdefault(y, []).append(f)

        return {
            cls: torch.stack(v).mean(dim=0)
            for cls, v in feats_by_class.items()
        }

    @torch.no_grad()
    def prototype_distance_scores(
        self,
        loader: DataLoader,
        prototypes: Dict[int, torch.Tensor],
    ):
        self.model.eval()

        proto_labels = list(prototypes.keys())
        proto_matrix = torch.stack(
            [prototypes[k] for k in proto_labels]
        ).to(self.device)
        proto_matrix = F.normalize(proto_matrix, dim=1)

        distances, labels = [], []

        for images, y_super, y_sub, _ in loader:
            images = images.to(self.device)
            feats = F.normalize(
                self.model.extract_features(images), dim=1
            )
            sim = feats @ proto_matrix.T
            dist = 1.0 - sim
            min_dist, _ = dist.min(dim=1)

            distances.extend(min_dist.cpu().tolist())
            labels.extend(y_sub.cpu().tolist())

        return distances, labels, proto_labels

    @torch.no_grad()
    def calibrate_prototype_threshold(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        prototypes: Dict[int, torch.Tensor],
        id_quantile: float = 0.8,
        ood_quantile: float = 0.2,
    ) -> Dict[str, float]:
        id_dist, _, _ = self.prototype_distance_scores(
            id_loader, prototypes
        )
        ood_dist, _, _ = self.prototype_distance_scores(
            ood_loader, prototypes
        )

        id_dist = torch.tensor(id_dist)
        ood_dist = torch.tensor(ood_dist)

        tau = float(
            (torch.quantile(id_dist, id_quantile)
             + torch.quantile(ood_dist, ood_quantile)) / 2
        )

        self.proto_tau = tau

        return {
            "tau_prototype": tau,
            "id_quantile": id_quantile,
            "ood_quantile": ood_quantile,
            "distance": "cosine",
        }

    # ---------------------------------------------------------
    # PROTOTYPE OSR
    # ---------------------------------------------------------
    @torch.no_grad()
    def evaluate_open_set_prototype(
        self,
        dataloader: DataLoader,
        prototypes_sub: Dict[int, torch.Tensor],
        prototypes_super: Dict[int, torch.Tensor],
        delta_sub: float,
        delta_super: float,
    ) -> Dict[str, float]:
        self.model.eval()

        def _run_level(prototypes, delta, is_sub):
            proto_labels = list(prototypes.keys())
            proto_mat = torch.stack(
                [prototypes[k] for k in proto_labels]
            ).to(self.device)
            proto_mat = F.normalize(proto_mat, dim=1)

            total = correct = 0
            seen = unseen = 0
            seen_correct = unseen_correct = 0

            for images, y_super, y_sub, _ in dataloader:
                images = images.to(self.device)
                labels = y_sub if is_sub else y_super
                labels = labels.to(self.device)

                feats = F.normalize(
                    self.model.extract_features(images), dim=1
                )
                dist = 1.0 - feats @ proto_mat.T
                min_dist, idx = dist.min(dim=1)

                preds = torch.tensor(
                    proto_labels, device=self.device
                )[idx]
                preds[min_dist > delta] = (
                    NOVEL_SUB_IDX if is_sub else NOVEL_SUPER_IDX
                )

                for p, y in zip(preds, labels):
                    total += 1
                    is_unseen = y.item() in (
                        NOVEL_SUB_IDX,
                        NOVEL_SUPER_IDX,
                    )
                    correct += int(p == y)
                    if is_unseen:
                        unseen += 1
                        unseen_correct += int(p == y)
                    else:
                        seen += 1
                        seen_correct += int(p == y)

            return correct, seen, seen_correct, unseen, unseen_correct, total

        sc, ss, ssc, su, suc, total = _run_level(
            prototypes_super, delta_super, False
        )
        c, s, sc2, u, uc, _ = _run_level(
            prototypes_sub, delta_sub, True
        )

        return {
            "super_overall": 100 * sc / total,
            "super_seen": 100 * ssc / max(1, ss),
            "super_unseen": 100 * suc / max(1, su),
            "sub_overall": 100 * c / total,
            "sub_seen": 100 * sc2 / max(1, s),
            "sub_unseen": 100 * uc / max(1, u),
        }

    # ---------------------------------------------------------
    # FUSED OSR
    # ---------------------------------------------------------
    @torch.no_grad()
    def evaluate_open_set_fused(
        self,
        dataloader,
        prototypes_super,
        prototypes_sub,
    ):
        """
        AND-voting OSR:
        - Superclass: confidence-only
        - Subclass: confidence AND prototype distance
        """
        self.model.eval()

        def _run_level(prototypes, is_sub):
            proto_labels = list(prototypes.keys())
            proto_mat = torch.stack([prototypes[k] for k in proto_labels]).to(self.device)
            proto_mat = F.normalize(proto_mat, dim=1)

            total = correct = 0
            seen = unseen = 0
            seen_correct = unseen_correct = 0

            tau_conf = self.tau_sub if is_sub else self.tau_super
            tau_proto = self.proto_tau if is_sub else None
            NOVEL_IDX = NOVEL_SUB_IDX if is_sub else NOVEL_SUPER_IDX

            for images, y_super, y_sub, _ in dataloader:
                images = images.to(self.device)
                labels = (y_sub if is_sub else y_super).to(self.device)

                logits_super, logits_sub = self.model(images)

                # ---------- Confidence ----------
                probs = torch.softmax(
                    logits_sub if is_sub else logits_super, dim=1
                )
                conf = probs.max(dim=1).values
                novel_conf = conf < tau_conf

                # ---------- Prototype ----------
                feats = F.normalize(self.model.extract_features(images), dim=1)
                dist = 1.0 - feats @ proto_mat.T
                min_dist, idx = dist.min(dim=1)

                proto_preds = torch.tensor(proto_labels, device=self.device)[idx]

                # ---------- Novel decision ----------
                if is_sub:
                    novel = novel_conf & (min_dist > tau_proto)
                else:
                    novel = novel_conf  # superclass = confidence only

                preds = proto_preds.clone()
                preds[novel] = NOVEL_IDX

                # ---------- Metrics ----------
                for p, y in zip(preds, labels):
                    total += 1
                    is_unseen = y.item() == NOVEL_IDX
                    correct += int(p == y)

                    if is_unseen:
                        unseen += 1
                        unseen_correct += int(p == y)
                    else:
                        seen += 1
                        seen_correct += int(p == y)

            return correct, seen, seen_correct, unseen, unseen_correct, total

        sc, ss, ssc, su, suc, total = _run_level(
            prototypes_super, is_sub=False
        )
        c, s, sc2, u, uc, _ = _run_level(
            prototypes_sub, is_sub=True
        )

        return {
            "super_overall": 100 * sc / total,
            "super_seen": 100 * ssc / max(1, ss),
            "super_unseen": 100 * suc / max(1, su),
            "sub_overall": 100 * c / total,
            "sub_seen": 100 * sc2 / max(1, s),
            "sub_unseen": 100 * uc / max(1, u),
        }



    # ---------------------------------------------------------
    # LOGGING
    # ---------------------------------------------------------
    def _append_epoch_metrics(self, record: dict) -> None:
        if self.run_path is None:
            return
        if not os.path.exists(self.run_path):
            raise FileNotFoundError(self.run_path)

        with open(self.run_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload.setdefault("training", {})
        payload["training"].setdefault("epochs", [])
        payload["training"]["epochs"].append(record)

        tmp = self.run_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.run_path)

    # ---------------------------------------------------------
    # EPOCH LOOP
    # ---------------------------------------------------------
    def _run_epoch(self, loader, train: bool) -> Dict[str, float]:
        totals = {
            "loss": 0.0,
            "loss_super": 0.0,
            "loss_sub": 0.0,
            "super_accuracy": 0.0,
            "sub_accuracy": 0.0,
            "count": 0,
        }

        pbar = tqdm(
            loader,
            desc="Train" if train else "Val",
            leave=False,
        )

        for batch in pbar:
            stats = self._step(batch, train)
            bs = stats["count"]

            for k in totals:
                if k != "count":
                    totals[k] += stats[k] * bs
            totals["count"] += bs

            pbar.set_postfix(
                loss=f"{totals['loss']/totals['count']:.4f}",
                super_acc=f"{totals['super_accuracy']/totals['count']:.3f}",
                sub_acc=f"{totals['sub_accuracy']/totals['count']:.3f}",
            )

        if train and self.scheduler is not None:
            self.scheduler.step()

        return {
            k: totals[k] / totals["count"]
            for k in totals
            if k != "count"
        }

    def train_epoch(self):
        self.model.train()
        return self._run_epoch(self.train_loader, True)

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        return self._run_epoch(self.val_loader, False)

    @torch.no_grad()
    def test_epoch(self):
        if self.test_loader is None:
            raise ValueError("No test_loader provided")
        self.model.eval()
        return self._run_epoch(self.test_loader, False)

    # ---------------------------------------------------------
    # FIT
    # ---------------------------------------------------------
    def fit(
        self,
        epochs: int,
        phase: str = "Epoch",
        early_stopper: Optional[EarlyStopper] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ):
        best_val = float("inf")

        for epoch in range(epochs):
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch()

            if verbose:
                print(
                    f"[{phase} {epoch+1:02d}] "
                    f"train_loss={train_stats['loss']:.4f}, "
                    f"val_loss={val_stats['loss']:.4f}, "
                    f"val_super_acc={val_stats['super_accuracy']:.4f}, "
                    f"val_sub_acc={val_stats['sub_accuracy']:.4f}"
                )

            if early_stopper and early_stopper.update(val_stats["loss"]):
                break

            if save_path and val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                torch.save(self.model.state_dict(), save_path)

            self._append_epoch_metrics(
                {
                    "epoch": epoch + 1,
                    "phase": phase,
                    "train": train_stats,
                    "val": val_stats,
                    "lr": (
                        self.optimizer.param_groups[0]["lr"]
                        if self.optimizer
                        else None
                    ),
                }
            )

    # ---------------------------------------------------------
    # CONFIDENCE OSR
    # ---------------------------------------------------------
    @torch.no_grad()
    def _collect_confidence(self, loader):
        super_scores, sub_scores = [], []

        for images, _, _, _ in loader:
            images = images.to(self.device)
            logits_super, logits_sub = self.model(images)

            super_scores.extend(
                F.softmax(logits_super, dim=1)
                .max(dim=1)
                .values.cpu()
            )
            sub_scores.extend(
                F.softmax(logits_sub, dim=1)
                .max(dim=1)
                .values.cpu()
            )

        return torch.stack(super_scores), torch.stack(sub_scores)

    @torch.no_grad()
    def calibrate(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        id_quantile: float = 0.2,
        ood_quantile: float = 0.8,
    ) -> Dict[str, float]:
        id_super, id_sub = self._collect_confidence(id_loader)
        ood_super, ood_sub = self._collect_confidence(ood_loader)

        self.tau_super = float(
            (torch.quantile(id_super, id_quantile)
             + torch.quantile(ood_super, ood_quantile)) / 2
        )
        self.tau_sub = float(
            (torch.quantile(id_sub, id_quantile)
             + torch.quantile(ood_sub, ood_quantile)) / 2
        )

        return {
            "tau_super": self.tau_super,
            "tau_sub": self.tau_sub,
        }

    @torch.no_grad()
    def evaluate_open_set(self, dataloader: DataLoader) -> Dict[str, float]:
        ce_loss = torch.nn.CrossEntropyLoss(reduction="sum")

        ce_super = ce_sub = 0.0
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

            probs_super = F.softmax(logits_super, dim=1)
            probs_sub = F.softmax(logits_sub, dim=1)

            conf_super, pred_super = probs_super.max(dim=1)
            conf_sub, pred_sub = probs_sub.max(dim=1)

            pred_super[conf_super < self.tau_super] = NOVEL_SUPER_IDX
            pred_sub[conf_sub < self.tau_sub] = NOVEL_SUB_IDX
            pred_sub[pred_super == NOVEL_SUPER_IDX] = NOVEL_SUB_IDX

            for i in range(len(y_super)):
                total += 1

                super_correct += int(pred_super[i] == y_super[i])
                sub_correct += int(pred_sub[i] == y_sub[i])

                if y_super[i] == NOVEL_SUPER_IDX:
                    unseen_super += 1
                    unseen_super_correct += int(
                        pred_super[i] == y_super[i]
                    )
                else:
                    seen_super += 1
                    seen_super_correct += int(
                        pred_super[i] == y_super[i]
                    )

                if y_sub[i] == NOVEL_SUB_IDX:
                    unseen_sub += 1
                    unseen_sub_correct += int(
                        pred_sub[i] == y_sub[i]
                    )
                else:
                    seen_sub += 1
                    seen_sub_correct += int(
                        pred_sub[i] == y_sub[i]
                    )

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

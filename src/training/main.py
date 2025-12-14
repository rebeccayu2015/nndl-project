import argparse
import json
from pathlib import Path
import torch
import os
from datetime import datetime
from typing import Any, Dict
from torch.utils.data import DataLoader, ConcatDataset

from src.training.configs import TrainingConfig
from src.training.trainer import (
    build_dataloaders,
    build_model,
    build_optimizer,
    build_scheduler,
    Trainer,
    EarlyStopper,
)
from src.utils.seed import set_seed


# -------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training entrypoint")

    parser.add_argument(
        "--mode",
        required=True,
        choices=["resnet50", "densenet121", "efficientnet_b0", "clip_b32"],
    )

    parser.add_argument("--train_csv", default="data/splits/train_split.csv")
    parser.add_argument("--val_csv", default="data/splits/val_split.csv")
    parser.add_argument("--calib_csv", default="data/splits/calibration_split.csv")
    parser.add_argument("--images_root", default="data/raw/train_images")
    parser.add_argument("--far_ood_root", default="data/raw/farood_images")
    parser.add_argument("--superclass_map", default="data/meta/superclass_mapping.csv")
    parser.add_argument("--subclass_map", default="data/meta/subclass_mapping.csv")

    parser.add_argument("--num_super", type=int, default=4)
    parser.add_argument("--num_sub", type=int, default=88)
    parser.add_argument("--lambda_sub", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--epochs_finetune", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--fine_tune", action="store_true", default=True)
    parser.add_argument("--no_fine_tune", action="store_false", dest="fine_tune")

    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--optimizer_finetune", choices=["adam", "adamw", "sgd"], default="adamw"
    )

    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument(
        "--scheduler_finetune", choices=["cosine", "none"], default="cosine"
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_finetune", type=float, default=1e-5)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--osr_only",
        action="store_true",
        help="Skip training and only run calibration + OSR using existing checkpoints",
    )

    return parser


# -------------------------------------------------------------
# JSON helpers
# -------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    try:
        if isinstance(obj, torch.device):
            return str(obj)
    except Exception:
        pass

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    return str(obj)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    os.replace(tmp_path, path)


# -------------------------------------------------------------
# Run artifact initialization
# -------------------------------------------------------------
def init_run_artifact(args, save_dir: str = "checkpoints", suffix: str = "") -> None:
    os.makedirs(save_dir, exist_ok=True)

    mode = getattr(args, "mode", None) or "run"

    if not hasattr(args, "run_id") or args.run_id is None:
        args.run_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    name = f"{args.run_id}{suffix}_run.json"
    args.run_path = os.path.join(save_dir, name)

    hparams = _json_safe(
        {k: v for k, v in vars(args).items() if k not in ("run_id", "run_path")}
    )

    existing = _load_json(args.run_path)

    base = {
        "run_id": args.run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": mode,
        "hyperparameters": hparams,
        "training": {"epochs": []},
        "osr": None,
    }

    if existing:
        existing.setdefault("run_id", base["run_id"])
        existing.setdefault("timestamp", base["timestamp"])
        existing.setdefault("model", base["model"])
        existing["hyperparameters"] = hparams
        existing.setdefault("training", {}).setdefault("epochs", [])
        existing.setdefault("osr", None)
        payload = existing
    else:
        payload = base

    _atomic_write_json(args.run_path, payload)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    suffix = "_osr" if args.osr_only else ""
    init_run_artifact(args, save_dir=args.save_dir, suffix=suffix)

    print(f"[INFO] Run ID: {args.run_id}")
    print(f"[INFO] Run JSON: {args.run_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    cfg = TrainingConfig(
        mode=args.mode,
        backbone=None if args.mode == "baseline" else args.mode,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        calib_csv=args.calib_csv,
        images_root=args.images_root,
        far_ood_root=args.far_ood_root,
        superclass_map=args.superclass_map,
        subclass_map=args.subclass_map,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        epochs_finetune=args.epochs_finetune,
        patience=args.patience,
        lr=args.lr,
        lr_finetune=args.lr_finetune,
        fine_tune=args.fine_tune,
        lambda_sub=args.lambda_sub,
        num_super=args.num_super,
        num_sub=args.num_sub,
        num_workers=args.num_workers,
        save_dir=f"{args.save_dir}/{args.mode}",
        seed=args.seed,
        optimizer=args.optimizer,
        optimizer_finetune=args.optimizer_finetune,
        scheduler=None if args.scheduler == "none" else args.scheduler,
        scheduler_finetune=None
        if args.scheduler_finetune == "none"
        else args.scheduler_finetune,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_head_path = save_dir / "best_head.pth"
    best_ft_path = save_dir / "best_finetune.pth"

    print("building dataloaders...")
    train_loader, val_loader, calib_loader, near_ood_loader, far_ood_loader = (
        build_dataloaders(cfg)
    )

    def collate_ignore_meta(batch):
        images, y_super, y_sub, _ = zip(*batch)
        return (
            torch.stack(images),
            torch.tensor(y_super),
            torch.tensor(y_sub),
            None,
        )

    open_set_dataset = ConcatDataset(
        [
            val_loader.dataset,
            near_ood_loader.dataset,
            far_ood_loader.dataset,
        ]
    )

    open_set_loader = DataLoader(
        open_set_dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        collate_fn=collate_ignore_meta,
    )

    print("building model and trainer...")
    model = build_model(cfg).to(device)

    trainer = Trainer(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lambda_sub=cfg.lambda_sub,
        patience=cfg.patience,
        run_path=args.run_path,
    )

    # ---------------- OSR-only ----------------
    if args.osr_only:
        print("[INFO] OSR-only mode: loading checkpoints and running calibration + OSR")

        if not best_head_path.exists():
            raise FileNotFoundError(f"Missing {best_head_path}")

        state = torch.load(best_head_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        trainer.calibrate(calib_loader, far_ood_loader)

        prototypes_sub = trainer.compute_prototypes(calib_loader, level="sub")
        prototypes_super = trainer.compute_prototypes(calib_loader, level="super")

        proto_calib_sub = trainer.calibrate_prototype_threshold(
            calib_loader, far_ood_loader, prototypes_sub
        )
        proto_calib_super = trainer.calibrate_prototype_threshold(
            calib_loader, far_ood_loader, prototypes_super
        )

        if cfg.fine_tune:
            if not best_ft_path.exists():
                raise FileNotFoundError(f"Missing {best_ft_path}")
            state = torch.load(best_ft_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            trainer.calibrate(calib_loader, far_ood_loader)

        metrics_conf = trainer.evaluate_open_set(open_set_loader)

        metrics_proto = trainer.evaluate_open_set_prototype(
            open_set_loader,
            prototypes_sub,
            prototypes_super,
            proto_calib_sub["tau_prototype"],
            proto_calib_super["tau_prototype"],
        )

        metrics_fused = trainer.evaluate_open_set_fused(
            open_set_loader,
            prototypes_super,
            prototypes_sub,
        )

        payload = _load_json(args.run_path)
        payload["osr"] = {
            "confidence": metrics_conf,
            "prototype_cosine": metrics_proto,
            "fused": {
                "metrics": metrics_fused,
            },
        }

        _atomic_write_json(args.run_path, payload)
        print("[INFO] OSR-only done.")
        return

    # ---------------- Training ----------------
    print("Start Training...")
    trainer.optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    trainer.scheduler = build_scheduler(cfg.scheduler, trainer.optimizer, cfg.num_epochs)
    stopper = EarlyStopper(cfg.patience)

    trainer.fit(cfg.num_epochs, "Epoch", stopper, best_head_path)
    trainer.calibrate(calib_loader, far_ood_loader)

    if not cfg.fine_tune:
        metrics_conf = trainer.evaluate_open_set(open_set_loader)
        payload = _load_json(args.run_path)
        payload.setdefault("osr", {})["confidence"] = metrics_conf
        _atomic_write_json(args.run_path, payload)
        return

    for p in model.backbone.parameters():
        p.requires_grad = True

    trainer.optimizer = build_optimizer(
        cfg.optimizer_finetune, model.parameters(), cfg.lr_finetune
    )
    trainer.scheduler = build_scheduler(
        cfg.scheduler_finetune, trainer.optimizer, cfg.epochs_finetune
    )
    stopper = EarlyStopper(cfg.patience)

    print("Start Fine Tuning...")
    trainer.fit(cfg.epochs_finetune, "Tune", stopper, best_ft_path)
    trainer.calibrate(calib_loader, far_ood_loader)

    metrics_conf = trainer.evaluate_open_set(open_set_loader)

    prototypes_sub = trainer.compute_prototypes(calib_loader, level="sub")
    prototypes_super = trainer.compute_prototypes(calib_loader, level="super")

    proto_calib_sub = trainer.calibrate_prototype_threshold(
        calib_loader, far_ood_loader, prototypes_sub
    )
    proto_calib_super = trainer.calibrate_prototype_threshold(
        calib_loader, far_ood_loader, prototypes_super
    )

    metrics_proto = trainer.evaluate_open_set_prototype(
        open_set_loader,
        prototypes_sub,
        prototypes_super,
        proto_calib_sub["tau_prototype"],
        proto_calib_super["tau_prototype"],
    )

    metrics_fused = trainer.evaluate_open_set_fused(
        open_set_loader,
        prototypes_super,
        prototypes_sub,
    )

    payload = _load_json(args.run_path)
    payload.setdefault("osr", {})
    payload["osr"]["confidence"] = metrics_conf
    payload["osr"]["prototype_cosine"] = metrics_proto
    payload["osr"]["fused"] = {
        "metrics": metrics_fused,
    }

    _atomic_write_json(args.run_path, payload)


if __name__ == "__main__":
    main()

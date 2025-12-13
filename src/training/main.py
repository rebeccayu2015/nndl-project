import argparse
import json
from pathlib import Path
import torch

from src.training.configs import TrainingConfig
from src.training.trainer import (
    build_dataloaders,
    build_model,
    build_optimizer,
    build_scheduler,
    Trainer,
    EarlyStopper,
)
from src.utils.logging import save_metrics
from src.utils.seed import set_seed

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
    parser.add_argument("--num_epochs", type=int, default=20) 
    parser.add_argument("--epochs_finetune", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--fine_tune", action="store_true", default=True)
    parser.add_argument("--no_fine_tune", action="store_false", dest="fine_tune")

    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--optimizer_finetune", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--scheduler_finetune", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_finetune", type=float, default=1e-5)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    args = build_parser().parse_args()

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
        scheduler_finetune=None if args.scheduler_finetune == "none" else args.scheduler_finetune,
    )

    # prepare dataloader 
    print("building dataloaders...")
    train_loader, val_loader, calib_loader, near_ood_loader, far_ood_loader = build_dataloaders(cfg)
    
    def collate_ignore_meta(batch):
        images, y_super, y_sub, _ = zip(*batch)
        return (
            torch.stack(images),
            torch.tensor(y_super),
            torch.tensor(y_sub),
            None,   # meta dropped
        )
        
    open_set_dataset = ConcatDataset([
        val_loader.dataset,
        near_ood_loader.dataset,
        far_ood_loader.dataset,
    ])
    
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
        optimizer=None,  # set per phase
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lambda_sub=cfg.lambda_sub,
        patience=cfg.patience,
    )

    # Stage 1: baseline or frozen backbone head training
    print("Start Training...")
    trainer.optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    trainer.scheduler = build_scheduler(cfg.scheduler, trainer.optimizer, cfg.num_epochs)

    stopper = EarlyStopper(cfg.patience)
    
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_head_path = save_dir / "best_head.pth"

    trainer.fit(cfg.num_epochs, "Epoch", stopper,best_head_path)
    trainer.calibrate(calib_loader, far_ood_loader)

    if not cfg.fine_tune:
        metrics = trainer.evaluate_open_set(open_set_loader)
        save_metrics(save_dir / "metric_output.csv", cfg.mode, metrics)
        return

    # Stage 2: fine-tune backbone
    for p in model.backbone.parameters():
        p.requires_grad = True

    trainer.optimizer = build_optimizer(cfg.optimizer_finetune, model.parameters(), cfg.lr_finetune)
    trainer.scheduler = build_scheduler(cfg.scheduler_finetune, trainer.optimizer, cfg.epochs_finetune)

    stopper = EarlyStopper(cfg.patience)

    best_ft_path = save_dir / "best_finetune.pth"
    
    print("Start Fine Tuning...")
    trainer.fit(cfg.epochs_finetune, "Tune", stopper, best_ft_path)
    trainer.calibrate(calib_loader, far_ood_loader)
    
    metrics = trainer.evaluate_open_set(open_set_loader)
    save_metrics(save_dir / "metric_output.csv", cfg.mode, metrics)

if __name__ == "__main__":
    main()

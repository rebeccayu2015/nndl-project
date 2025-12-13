from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    mode: str
    train_csv: str
    val_csv: str
    calib_csv: str
    images_root: str
    far_ood_root: str
    superclass_map: str
    subclass_map: str
    num_super: int
    num_sub: int
    lambda_sub: float
    batch_size: int
    num_epochs: int
    lr: float
    num_workers: int
    save_dir: str
    seed: int
    backbone: Optional[str] = None
    fine_tune: bool = True
    optimizer: str = "adamw"
    optimizer_finetune: str = "adamw"
    scheduler: Optional[str] = None  
    scheduler_finetune: Optional[str] = None
    lr_finetune: Optional[float] = None
    epochs_finetune: Optional[int] = None
    patience: Optional[int] = None

from src.data.data_utils import export_cifar100

 export_cifar100(
    out_dir="data/raw/farood_images",
    train=False,
    max_items=5000,
    clear_dir=True
 )
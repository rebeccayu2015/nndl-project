from src.data.data_utils import create_splits

create_splits(
    "data/meta/train_data.csv",
    "data/splits",
    val_size=0.15,
    calib_size_from_train=0.10,
)

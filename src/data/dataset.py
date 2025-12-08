"""
dataset.py
-----------

Defines the custom PyTorch Dataset class used throughout the project

1. Loads image paths and label information from the provided CSV files
   (train_data.csv, train/val/calibration splits, superclass/subclass mappings).

2. Reads images from disk (using PIL) and applies the appropriate
   preprocessing/augmentation transforms passed in from transforms.py.

3. Returns a structured sample for each index:
    - image tensor (after transforms)
    - superclass label (int)
    - subclass label (int)
    - optional metadata (filename, original paths, etc.)
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BirdDogReptileDataset(Dataset):
    """
    Returns (image, y_super, y_sub, meta_dict)
    where y_super in {0,1,2} (bird/dog/reptile) and y_sub in [0..num_sub-1].
    """
    def __init__(self,
                 csv_path: str,
                 images_root: str,
                 transform=None,
                 superclass_mapping_path: str = None,
                 subclass_mapping_path: str = None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform = transform

        self.superclass_mapping = None
        self.subclass_mapping = None

        if superclass_mapping_path is not None:
            self.superclass_mapping = pd.read_csv(superclass_mapping_path)
        if subclass_mapping_path is not None:
            self.subclass_mapping = pd.read_csv(subclass_mapping_path)

        # Basic sanity checks
        assert 'image' in self.df.columns
        assert 'superclass_index' in self.df.columns
        assert 'subclass_index' in self.df.columns

    def __len__(self):
        return len(self.df)

    def _load_image(self, idx: int) -> Image.Image:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row['image'])
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(idx)
        if self.transform is not None:
            img = self.transform(img)

        y_super = int(row['superclass_index'])
        y_sub = int(row['subclass_index'])

        meta = {
            "image": row['image']
        }
        if 'description' in row:
            meta["description"] = row['description']

        return img, y_super, y_sub, meta

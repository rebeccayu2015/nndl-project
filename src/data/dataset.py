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
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.utils.const import NOVEL_SUPER_IDX, NOVEL_SUB_IDX

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
        
class TestDataset(Dataset):
    def __init__(self,
                 images_root: str,
                 transform=None,
                 superclass_mapping_path: str = None,
                 subclass_mapping_path: str = None):
        self.images_root = images_root
        self.transform = transform

        self.superclass_mapping = None
        self.subclass_mapping = None

        if superclass_mapping_path is not None:
            self.superclass_mapping = pd.read_csv(superclass_mapping_path)
        if subclass_mapping_path is not None:
            self.subclass_mapping = pd.read_csv(subclass_mapping_path)

    def __len__(self):
        return len([fname for fname in os.listdir(self.images_root) if '.jpg' in fname])

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        img_path = os.path.join(self.images_root, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name
        
class NearOODDataset(Dataset):
    """
    Near-OOD dataset created by applying strong corruptions
    to in-distribution images.

    Labels are forced to NOVEL for calibration purposes.
    """
    def __init__(self, 
                 base_dataset: str,  
                 transform=None):
        """
        base_dataset: BirdDogReptileDataset with transform=None
        """
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base._load_image(idx)
        row = self.base.df.iloc[idx]
        if self.transform is not None:
            img = self.transform(img)
            
        meta = {
            "image": row["image"],
            "source": "near_ood",
        }

        return img, NOVEL_SUPER_IDX, NOVEL_SUB_IDX, meta


class FarOODDataset(Dataset):
    """
    Far-OOD dataset loaded from a local directory of .jpg images.

    Returns:
        (image_tensor, NOVEL_SUPER_IDX, NOVEL_SUB_IDX, meta)
    """
    def __init__(self, 
                 images_root: str, 
                 transform=None):
        self.images_root = images_root
        self.transform = transform

        self.paths = sorted(glob.glob(os.path.join(images_root, "*.jpg")))
        
        if len(self.paths) == 0:
            raise ValueError(f"No .jpg images found in: {images_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        meta = {
            "image": os.path.basename(path),
            "source": "far_ood",
        }
        return img, NOVEL_SUPER_IDX, NOVEL_SUB_IDX, meta

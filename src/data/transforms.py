"""
transforms.py
--------------

Defines all image preprocessing and data augmentation pipelines used
throughout the project

These transforms serve two main purposes:

1. Preprocessing
   - Convert raw PIL images into PyTorch tensors
   - Normalize images using ImageNet mean and std so that pretrained
     backbones (EfficientNet, ResNet, DenseNet) behave correctly
   - Ensure consistent input size (64×64 in this project)

2. Data Augmentation (training only)
   - Random horizontal flips
   - Small rotations
   - Random resized crops
   - Mild color jitter
   These augmentations improve the model's robustness and ability to
   generalize to distribution shifts, including unseen subclasses.

Functions return torchvision `transforms.Compose` objects which can be
passed directly into the Dataset class.
"""

from torchvision import transforms

# Can adjust after computing dataset mean/std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

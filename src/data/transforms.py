"""
transforms.py
--------------

Defines all image preprocessing (i.e. normalization) and data augmentation pipelines used
throughout the project

1. Preprocessing
   - Convert raw PIL images into PyTorch tensors
   - Normalize images using ImageNet mean and std so that pretrained
     backbones (EfficientNet, ResNet, DenseNet) behave correctly
   - Ensure consistent input size 

2. Data Augmentation (training only)
   - Random horizontal flips
   - Small rotations
   - Random resized crops
   - Mild color jitter

Functions return torchvision `transforms.Compose` objects which can be
passed directly into the Dataset class.
"""

import torchvision.transforms as T
import clip

# Can adjust after computing dataset mean/std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ResNet50 / DenseNet121 / EfficientNet-B0 were pretrained with ~224×224 inputs and ImageNet normalization. 
# Using 224×224 usually gives best results.

def get_train_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_clip_transform():
    _, preprocess = clip.load("ViT-B/32", device="cpu")
    return preprocess
    
def get_near_ood_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.03, 0.15)),  # extreme crop
        T.RandomGrayscale(p=1.0),
        T.GaussianBlur(kernel_size=23),
        T.ColorJitter(brightness=2.0, contrast=2.0, saturation=2.0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

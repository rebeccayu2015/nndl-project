"""
baseline_cnn.py
----------------

Defines the simple convolutional neural network used as the baseline
model for this project

Simple CNN with shared conv backbone and two heads:
    - superclass head: 3 classes (bird, dog, reptile)
    - subclass head: N subclasses (from train_data.csv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, num_superclasses: int = 3, num_subclasses: int = 87):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8x8
        )

        self.flatten_dim = 128 * 8 * 8

        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.super_head = nn.Linear(256, num_superclasses)
        self.sub_head = nn.Linear(256, num_subclasses)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        logits_super = self.super_head(x)
        logits_sub = self.sub_head(x)
        return logits_super, logits_sub

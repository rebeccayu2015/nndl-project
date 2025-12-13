# src/models/dual_head.py
"""
Dual-head classifier model for superclass + subclass prediction.

Uses a transfer-learning backbone (EfficientNet, ResNet50, DenseNet121, etc.)
built via build_backbone() and attaches two classification heads:

    - super_head : predicts superclass (num_super classes)
    - sub_head   : predicts subclass (num_sub classes)

Returns:
    logits_super, logits_sub
"""

import torch
import torch.nn as nn

from src.models.backbones import build_backbone   # <-- uses existing repo structure


class DualHeadNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_super: int = 4,
        num_sub: int = 88,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        
    ):
        super().__init__()

        # build_backbone must return (model, feature_dim)
        self.backbone, feat_dim = build_backbone(
            backbone_name, 
            pretrained=pretrained
        )

        # Two classifier heads
        self.super_head = nn.Linear(feat_dim, num_super)
        self.sub_head   = nn.Linear(feat_dim, num_sub)

        # freeze backbone parameters
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)           # (B, 2048, 1, 1)
        feats = torch.flatten(feats, 1)    # (B, 2048)
        # print("BACKBONE OUTPUT SHAPE:", feats.shape)
        logits_super = self.super_head(feats)
        logits_sub   = self.sub_head(feats)
        return logits_super, logits_sub
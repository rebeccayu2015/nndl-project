"""
eval.py
--------

Evaluation utilities for computing metrics used throughout the project

Metrics:
    - Superclass accuracy: computes accuracy over the three superclasses (bird, dog, reptile)
    - Subclass accuracy: computes accuracy over the 87 fine-grained subclasses
    - Combined or per-head cross-entropy loss
    - Seen vs. unseen accuracy: used during OSR evaluation, where known-class and novel-class accuracy must be measured separately
    - Utility helpers for aggregating and storing metrics in dictionaries or JSON files
"""

from typing import Dict, Optional
import torch
import torch.nn.functional as F

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: (N, C), targets: (N,)
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)

def compute_super_sub_accuracies(
    logits_super: torch.Tensor,
    logits_sub: torch.Tensor,
    y_super: torch.Tensor,
    y_sub: torch.Tensor,
) -> Dict[str, float]:
    super_acc = accuracy_from_logits(logits_super, y_super)
    sub_acc = accuracy_from_logits(logits_sub, y_sub)
    return {
        "super_accuracy": super_acc,
        "sub_accuracy": sub_acc,
    }

def cross_entropy_loss(
    logits_super: torch.Tensor,
    logits_sub: torch.Tensor,
    y_super: torch.Tensor,
    y_sub: torch.Tensor,
    lambda_sub: float = 1.0
) -> torch.Tensor:
    loss_super = F.cross_entropy(logits_super, y_super)
    loss_sub = F.cross_entropy(logits_sub, y_sub)
    return loss_super + lambda_sub * loss_sub

# Only for OSR
def compute_seen_unseen_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    is_seen_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    is_seen_mask: Boolean tensor of shape (num_classes,),
      True for seen, False for unseen.
    """
    assert preds.shape == targets.shape
    correct = (preds == targets)

    seen_mask_for_targets = is_seen_mask[targets]
    unseen_mask_for_targets = ~seen_mask_for_targets

    seen_correct = correct[seen_mask_for_targets].sum().item()
    seen_total = seen_mask_for_targets.sum().item()

    unseen_correct = correct[unseen_mask_for_targets].sum().item()
    unseen_total = unseen_mask_for_targets.sum().item()

    seen_acc = seen_correct / seen_total if seen_total > 0 else 0.0
    unseen_acc = unseen_correct / unseen_total if unseen_total > 0 else 0.0

    overall_acc = correct.sum().item() / len(targets)

    return {
        "overall": overall_acc,
        "seen": seen_acc,
        "unseen": unseen_acc,
    }

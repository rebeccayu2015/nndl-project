"""
seed.py
-------

Utility for making experiments deterministic and reproducible

Fixes all relevant random number generators so that:
- model initialization is the same every run
- data shuffling is consistent
- augmentation randomness is reproducible
- training curves and results become comparable across runs
- experiments can be reproduced reliably
"""

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

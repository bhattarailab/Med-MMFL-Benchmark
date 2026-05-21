"""Reproducibility utilities for setting random seeds across all frameworks."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The random seed value to use across all random number generators.
        deterministic: If True, enables deterministic cuDNN behavior at the cost
            of potential performance reduction. Ensures identical convolution
            results across runs.

    Example:
        >>> set_seed(42)
        >>> torch.rand(3)  # Will produce the same result every time
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU support
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

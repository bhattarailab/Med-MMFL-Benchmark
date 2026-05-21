"""Loss function registry and implementations.

Provides a factory function ``get_criterion`` to construct loss functions
by name from configuration.
"""

import torch.nn as nn

from med_mmfl_bench.losses.dice_bce import DiceBCELoss, DiceLoss
from med_mmfl_bench.losses.symile_loss import clip, symile

__all__ = [
    "get_criterion",
    "DiceBCELoss",
    "DiceLoss",
    "clip",
    "symile",
]

_CRITERION_REGISTRY = {
    "dicebceloss": DiceBCELoss,
    "clip": clip,
    "bcewithlogitsloss": nn.BCEWithLogitsLoss,
    "symile": symile,
    "crossentropyloss": nn.CrossEntropyLoss,
}


def get_criterion(criterion_name: str) -> nn.Module:
    """Create a loss function by name.

    Args:
        criterion_name: Name of the loss function (case-insensitive).
            Supported: ``"DiceBCELoss"``, ``"clip"``, ``"BCEWithLogitsLoss"``.

    Returns:
        Loss function instance or callable.

    Raises:
        ValueError: If ``criterion_name`` is not recognized.
    """
    key = criterion_name.lower()
    if key not in _CRITERION_REGISTRY:
        raise ValueError(
            f"Unknown loss function: '{criterion_name}'. "
            f"Available: {list(_CRITERION_REGISTRY.keys())}"
        )

    factory = _CRITERION_REGISTRY[key]

    # Some entries are already callable functions (e.g., `clip`)
    if isinstance(factory, type):
        return factory()
    return factory

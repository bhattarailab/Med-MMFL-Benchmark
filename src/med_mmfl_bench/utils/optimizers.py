"""Optimizer and learning rate scheduler factory functions."""

import warnings
from typing import Any, Dict, Iterator, Optional

import torch.optim as optim
from torch.nn import Parameter

from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["get_optimizer", "get_lr_scheduler"]


def get_optimizer(
    optimizer_name: str,
    parameters: Iterator[Parameter],
    config: Any,
) -> optim.Optimizer:
    """Create an optimizer instance from a name and configuration.

    Args:
        optimizer_name: Name of the optimizer. Supported values (case-insensitive):
            ``"adam"``, ``"adamw"``, ``"sgd"``.
        parameters: Iterator over model parameters to optimize.
        config: Configuration object (``munch.Munch`` or dict-like) containing
            optimizer hyperparameters (``learning_rate``, ``weight_decay``, etc.).

    Returns:
        Configured PyTorch optimizer instance.

    Raises:
        ValueError: If ``optimizer_name`` is not a supported optimizer.

    Example:
        >>> optimizer = get_optimizer("adam", model.parameters(), config.optimizer)
    """
    name_lower = optimizer_name.lower()

    if name_lower == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=float(config.learning_rate),
            betas=config.get("betas", (0.9, 0.999)),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=float(config.get("weight_decay", 0)),
            amsgrad=config.get("amsgrad", False),
        )
    elif name_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            lr=float(config.learning_rate),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
    elif name_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=float(config.get("learning_rate", 0.001)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(
            f"Invalid optimizer name: '{optimizer_name}'. "
            f"Supported: 'adam', 'adamw', 'sgd'."
        )

    logger.debug("Created %s optimizer with config: %s", optimizer_name, config)
    return optimizer


def get_lr_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    config: Any,
) -> optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler from a name and configuration.

    Args:
        scheduler_name: Name of the scheduler. Supported values:
            ``"reduce_lr_on_plateau"``, ``"cosine_annealing"``.
        optimizer: The optimizer whose learning rate will be scheduled.
        config: Configuration object containing scheduler hyperparameters.

    Returns:
        Configured PyTorch learning rate scheduler.

    Raises:
        ValueError: If ``scheduler_name`` is not a supported scheduler.
    """
    name_lower = scheduler_name.lower()

    if name_lower == "reduce_lr_on_plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=float(config.get("factor", 0.1)),
            patience=config.get("patience", 10),
            threshold=float(config.get("threshold", 1e-4)),
            threshold_mode=config.get("threshold_mode", "rel"),
            cooldown=int(config.get("cooldown", 0)),
            min_lr=float(config.get("min_lr", 0)),
            eps=float(config.get("eps", 1e-8)),
        )
    elif name_lower == "cosine_annealing":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
        )
    else:
        raise ValueError(
            f"Invalid scheduler name: '{scheduler_name}'. "
            f"Supported: 'reduce_lr_on_plateau', 'cosine_annealing'."
        )

    logger.debug("Created %s scheduler with config: %s", scheduler_name, config)
    return lr_scheduler

"""Weight initialization utilities for neural network modules."""

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["init_weights"]


def init_weights(
    model: nn.Module,
    init_type: str,
    init_gain: float = 0.02,
) -> None:
    """Initialize network weights using the specified strategy.

    Applies the chosen initialization to all Conv and Linear layers. BatchNorm
    layers are always initialized with mean=1.0, bias=0.0.

    Args:
        model: Network module to initialize.
        init_type: Initialization method. One of: ``"normal"``, ``"xavier"``,
            ``"xavier_uniform"``, ``"kaiming"``, ``"truncnorm"``,
            ``"orthogonal"``, ``"none"`` (uses PyTorch defaults).
        init_gain: Scaling factor for normal, xavier, and orthogonal
            initializations.

    Raises:
        NotImplementedError: If ``init_type`` is not a recognized method.

    Example:
        >>> model = nn.Linear(10, 5)
        >>> init_weights(model, "xavier", init_gain=1.0)
    """
    _INIT_METHODS = {
        "normal": lambda w: nn.init.normal_(w, mean=0.0, std=init_gain),
        "xavier": lambda w: nn.init.xavier_normal_(w, gain=init_gain),
        "xavier_uniform": lambda w: nn.init.xavier_uniform_(w, gain=init_gain),
        "kaiming": lambda w: nn.init.kaiming_normal_(w, a=0, mode="fan_in"),
        "truncnorm": lambda w: nn.init.trunc_normal_(w, mean=0.0, std=init_gain),
        "orthogonal": lambda w: nn.init.orthogonal_(w, gain=init_gain),
    }

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__

        # BatchNorm layers
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        # Linear and Conv layers
        elif hasattr(m, "weight") and (
            classname.startswith("Linear") or classname.startswith("Conv")
        ):
            if init_type == "none":
                m.reset_parameters()
            elif init_type in _INIT_METHODS:
                _INIT_METHODS[init_type](m.weight.data)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented. "
                    f"Supported: {list(_INIT_METHODS.keys()) + ['none']}"
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)

"""Abstract base class for federated optimization algorithms."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch

__all__ = ["BaseOptimizer"]


class BaseOptimizer(metaclass=ABCMeta):
    """Abstract base class for federated aggregation strategies.

    Defines the interface for dispatching model parameters to clients and
    aggregating updated parameters back to the server.

    Attributes:
        role: Role identifier (``"server"`` or ``"client"``).
    """

    @abstractmethod
    def dispatch(self, *args: Any, **kwargs: Any) -> None:
        """Dispatch server model parameters to a client."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, *args: Any, **kwargs: Any) -> None:
        """Upload and aggregate client parameters to the server model."""
        raise NotImplementedError

    @staticmethod
    def average_weights(
        w: List[Dict[str, torch.Tensor]],
        omega: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted average of model state dictionaries.

        Args:
            w: List of model state dicts from participating clients.
            omega: Optional weight vector for each client. If None, uses
                uniform weighting.

        Returns:
            Averaged state dictionary with the same keys as the inputs.
        """
        if omega is None:
            omega = np.ones(len(w)) / len(w)
        else:
            omega = omega / omega.sum()

        w_avg: Dict[str, torch.Tensor] = {}
        for key in w[0].keys():
            w_avg[key] = sum(
                omega[i] * w[i][key].float() for i in range(len(w))
            )
        return w_avg

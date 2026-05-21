"""Abstract base class for federated learning servers."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import torch

__all__ = ["BaseServer"]


class BaseServer(metaclass=ABCMeta):
    """Abstract base class for federated learning servers.

    The central server orchestrates the federated learning process:
    1. Selects a subset of clients for each communication round.
    2. Broadcasts the global model to selected clients.
    3. Collects and aggregates updated model parameters.
    4. Evaluates the aggregated model.

    Attributes:
        device: Torch device for computation.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Execute one communication round of federated learning."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> Optional[Dict[str, float]]:
        """Evaluate the global model.

        Returns:
            Dictionary of evaluation metrics, or None.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, **kwargs: Any) -> None:
        """Finalize training (save best model, run test evaluation)."""
        raise NotImplementedError

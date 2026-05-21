"""Abstract base class for all trainers."""

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

__all__ = ["BaseTrainer"]


class BaseTrainer(metaclass=ABCMeta):
    """Abstract base class for dataset-specific trainers.

    Trainers encapsulate the full training loop including data loading,
    forward/backward passes, validation, and testing. They are used by
    both federated clients and standalone training scripts.

    Subclasses must implement ``run``, ``train``, ``val``, and ``test``.
    """

    @abstractmethod
    def run(self, **kwargs: Any) -> None:
        """Execute the full training pipeline."""
        raise NotImplementedError

    @abstractmethod
    def train(self, **kwargs: Any) -> None:
        """Execute one training epoch."""
        raise NotImplementedError

    @abstractmethod
    def val(self, **kwargs: Any) -> Optional[float]:
        """Execute validation and return the primary metric."""
        raise NotImplementedError

    @abstractmethod
    def test(self, **kwargs: Any) -> None:
        """Execute evaluation on the test set."""
        raise NotImplementedError

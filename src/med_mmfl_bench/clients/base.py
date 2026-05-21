"""Abstract base class for federated learning clients."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

import torch

__all__ = ["BaseClient"]


class BaseClient(metaclass=ABCMeta):
    """Abstract base class for federated learning clients.

    Clients participate in each communication round by:
    1. Downloading the global model from the server (``download``).
    2. Performing local training on private data (``update``).
    3. Uploading updated model parameters to the server (``upload``).

    Subclasses are expected to set ``client_id`` and ``device`` in
    their own ``__init__``.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Perform local training on private data."""
        raise NotImplementedError

    @abstractmethod
    def upload(self) -> Any:
        """Upload local model parameters to the server.

        Returns:
            State dictionary or algorithm-specific upload payload.
        """
        raise NotImplementedError

    @abstractmethod
    def download(self, model: Any) -> None:
        """Download the global model from the server.

        Args:
            model: Global model or state dictionary.
        """
        raise NotImplementedError

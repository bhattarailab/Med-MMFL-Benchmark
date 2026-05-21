"""Federated learning client implementations.

Each client encapsulates local training logic, model management, and
communication with the central server.
"""

from med_mmfl_bench.clients.base import BaseClient

__all__ = ["BaseClient"]

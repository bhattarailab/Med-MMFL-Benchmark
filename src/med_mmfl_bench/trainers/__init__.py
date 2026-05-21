"""Trainer implementations for different dataset modalities.

Each trainer encapsulates the training, validation, and evaluation logic
for a specific dataset type, decoupling it from the federated learning
orchestration layer.
"""

from med_mmfl_bench.trainers.base import BaseTrainer

__all__ = ["BaseTrainer"]

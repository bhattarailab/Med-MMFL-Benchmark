"""Custom collate functions for DataLoader batching.

Provides specialized collate functions for VQA-style datasets where
samples are dictionaries of tensors that need to be stacked by key.
"""

from typing import Any, Dict, List

import torch

__all__ = ["default_collate_fn", "vqa_collate_fn"]


def default_collate_fn(batch: List[Any]) -> List[Any]:
    """Pass-through collate that returns the batch list unchanged.

    Useful as a fallback when no special batching logic is needed.

    Args:
        batch: List of samples from the dataset.

    Returns:
        The batch list, unchanged.
    """
    return batch


def vqa_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for VQA datasets (PathVQA, EHRXQA).

    Stacks dictionary-style samples into a single batched dictionary.
    Expected keys: ``input_ids``, ``attention_mask``, ``pixel_values``,
    ``labels``.

    Args:
        batch: List of sample dictionaries, each containing tensor values.

    Returns:
        Batched dictionary with stacked tensors.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }
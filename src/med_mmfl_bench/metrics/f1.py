"""Per-class F1 score metric for multi-class classification."""

from typing import Dict, List

import numpy as np
import torch
from torchmetrics import Metric

__all__ = ["F1PerClass"]


class F1PerClass(Metric):
    """Compute per-class F1 scores over accumulated predictions.

    Accumulates prediction probability distributions and integer targets
    across batches, then computes per-class F1 and macro average at
    ``compute()`` time using TP/FP/FN counting.

    Args:
        classes: List of class names for the output dictionary.

    Example:
        >>> metric = F1PerClass(classes=["No", "Yes"])
        >>> metric.update(preds, targets)
        >>> result = metric.compute()
        >>> result["average"]
        0.82
    """

    def __init__(self, classes: List[str] = ("NO", "YES"), **kwargs) -> None:
        super().__init__(dist_sync_on_step=False)
        self.classes = list(classes)

        self.add_state("all_preds", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions and targets.

        Args:
            preds: Class probabilities or logits ``(N, num_classes)``.
            targets: Integer class labels ``(N,)``.
        """
        self.all_preds.append(preds.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute per-class F1 scores and macro average.

        Returns:
            Dictionary mapping class names to F1 scores, plus an
            ``"average"`` key for the macro F1. Classes with no TP/FP/FN
            are reported as ``NaN``.
        """
        preds = torch.cat(self.all_preds, dim=0)
        targets = torch.cat(self.all_targets, dim=0)

        predicted_classes = torch.argmax(preds, dim=1)

        scores: Dict[str, float] = {}
        for class_idx, class_name in enumerate(self.classes):
            tp = ((predicted_classes == class_idx) & (targets == class_idx)).sum().item()
            fp = ((predicted_classes == class_idx) & (targets != class_idx)).sum().item()
            fn = ((predicted_classes != class_idx) & (targets == class_idx)).sum().item()

            if tp + fp + fn == 0:
                scores[class_name] = float("nan")
            else:
                scores[class_name] = (2 * tp) / (2 * tp + fp + fn)

        valid_scores = [v for v in scores.values() if not np.isnan(v)]
        scores["average"] = (
            float(np.mean(valid_scores)) if valid_scores else float("nan")
        )
        return scores
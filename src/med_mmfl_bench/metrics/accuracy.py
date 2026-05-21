"""Per-class accuracy metric for multi-class classification."""

from typing import Dict, List

import numpy as np
import torch
from torchmetrics import Metric

__all__ = ["AccuracyPerClass"]


class AccuracyPerClass(Metric):
    """Compute per-class accuracy over accumulated predictions.

    Accumulates prediction probability distributions and integer targets
    across batches, then computes per-class accuracy and macro average
    at ``compute()`` time.

    Args:
        classes: List of class names for the output dictionary.

    Example:
        >>> metric = AccuracyPerClass(classes=["No", "Yes"])
        >>> metric.update(preds, targets)
        >>> result = metric.compute()
        >>> result["average"]
        0.85
    """

    def __init__(self, classes: List[str] = ("Class0", "Class1"), **kwargs) -> None:
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
        """Compute per-class accuracy and macro average.

        Returns:
            Dictionary mapping class names to accuracy scores, plus an
            ``"average"`` key for the macro average. Classes with no
            samples are reported as ``NaN``.
        """
        preds = torch.cat(self.all_preds, dim=0)
        targets = torch.cat(self.all_targets, dim=0)

        predicted_classes = torch.argmax(preds, dim=1)

        scores: Dict[str, float] = {}
        for class_idx, class_name in enumerate(self.classes):
            mask = targets == class_idx
            if mask.sum() == 0:
                scores[class_name] = float("nan")
            else:
                correct = (predicted_classes[mask] == targets[mask]).sum().item()
                total = mask.sum().item()
                scores[class_name] = correct / total

        valid_scores = [v for v in scores.values() if not np.isnan(v)]
        scores["average"] = (
            float(np.mean(valid_scores)) if valid_scores else float("nan")
        )
        return scores
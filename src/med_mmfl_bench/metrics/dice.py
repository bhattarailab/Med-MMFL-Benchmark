"""Per-class Dice score metric for segmentation evaluation."""

from typing import Dict, List

import numpy as np
import torch
from torchmetrics import Metric

__all__ = ["DicePerClass"]


class DicePerClass(Metric):
    """Compute per-class Dice scores over accumulated predictions.

    Accumulates predictions and targets across batches, then computes
    the Dice coefficient for each class at ``compute()`` time.

    Args:
        threshold: Binarization threshold for predictions.
        eps: Small constant to avoid division by zero.
        classes: List of class names for the output dictionary.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-9,
        classes: List[str] = ("WT", "TC", "ET"),
    ) -> None:
        super().__init__(dist_sync_on_step=False)
        self.threshold = threshold
        self.eps = eps
        self.classes = list(classes)

        self.add_state("all_preds", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions and targets.

        Args:
            preds: Model predictions ``(B, C, *spatial_dims)``.
            targets: Ground truth masks ``(B, C, *spatial_dims)``.
        """
        self.all_preds.append(preds.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute per-class Dice scores and macro average.

        Returns:
            Dictionary mapping class names to Dice scores, plus an
            ``"average"`` key for the macro average.
        """
        preds = torch.cat(self.all_preds, dim=0).numpy()
        targets = torch.cat(self.all_targets, dim=0).numpy()
        num_samples, num_classes = preds.shape[:2]
        predictions = (preds >= self.threshold).astype(np.float32)

        assert predictions.shape == targets.shape, (
            f"Shape mismatch: {predictions.shape} vs {targets.shape}"
        )

        scores: Dict[str, List[float]] = {key: [] for key in self.classes}

        for i in range(num_samples):
            for cls_idx in range(num_classes):
                p = predictions[i][cls_idx]
                t = targets[i][cls_idx]

                intersection = 2.0 * (t * p).sum()
                union = t.sum() + p.sum()

                if t.sum() == 0 and p.sum() == 0:
                    score = 1.0
                else:
                    score = float((intersection + self.eps) / (union + self.eps))
                scores[self.classes[cls_idx]].append(score)

        per_class_scores = {
            cls: float(np.mean(scores[cls])) for cls in self.classes
        }
        per_class_scores["average"] = float(
            np.mean(list(per_class_scores.values()))
        )
        return per_class_scores

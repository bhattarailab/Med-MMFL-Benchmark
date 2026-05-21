"""Dice and Dice-BCE combined loss functions for segmentation tasks."""

import torch
import torch.nn as nn

__all__ = ["DiceLoss", "DiceBCELoss"]


class DiceLoss(nn.Module):
    """Soft Dice loss for binary/multi-class segmentation.

    Computes 1 - DiceCoefficient, where the Dice coefficient measures
    the overlap between predicted probabilities and ground truth masks.

    Args:
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Raw model predictions of shape ``(B, C, *spatial_dims)``.
            targets: Ground truth masks of shape ``(B, C, *spatial_dims)``.

        Returns:
            Scalar Dice loss value.
        """
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)

        assert probability.shape == targets.shape, (
            f"Shape mismatch: predictions {probability.shape} vs "
            f"targets {targets.shape}"
        )

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / (union + self.eps)
        return 1.0 - dice_score


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for segmentation.

    Sums the Dice loss and ``BCEWithLogitsLoss`` for improved gradient
    flow during training.

    Args:
        smooth: Smoothing constant for Dice loss (passed as ``eps``).
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(eps=smooth)

    def forward(
        self,
        inputs: torch.Tensor,
        oh_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined Dice-BCE loss.

        Args:
            inputs: Raw model predictions ``(B, C, D, H, W)``.
            oh_targets: One-hot ground truth masks ``(B, C, D, H, W)``.

        Returns:
            Scalar combined loss value.
        """
        ce_loss = self.bce(inputs, oh_targets)
        dice_loss = self.dice_loss(inputs, oh_targets)
        return dice_loss + ce_loss

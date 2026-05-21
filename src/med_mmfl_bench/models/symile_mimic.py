"""SYMILE-MIMIC model for 3-modality contrastive learning.

Encodes CXR images, ECG signals, and lab results into a shared
embedding space for contrastive alignment.
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["SymileMIMICModel", "CXREncoder", "ECGEncoder", "LabsEncoder"]


class CXREncoder(nn.Module):
    """Chest X-ray encoder using ResNet-50 backbone.

    Encodes CXR images into a normalized d-dimensional representation
    using a ResNet-50 backbone with a replaced FC layer and LayerNorm.

    Args:
        config: Configuration with ``d`` (embedding dimension) and
            ``pretrained`` (whether to use ImageNet weights).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        if config.pretrained:
            self.resnet = models.resnet50(weights="IMAGENET1K_V2")
        else:
            self.resnet = models.resnet50(weights=None)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, config.d, bias=True)
        self.layer_norm = nn.LayerNorm(config.d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode CXR images.

        Args:
            x: CXR data ``(batch_sz, 3, 320, 320)``.

        Returns:
            Learned CXR representation ``(batch_sz, d)``.
        """
        x = self.resnet(x)
        return self.layer_norm(x)


class ECGEncoder(nn.Module):
    """ECG signal encoder using ResNet-18 backbone.

    Encodes single-channel ECG signals into a normalized d-dimensional
    representation using a modified ResNet-18 backbone.

    Args:
        config: Configuration with ``d`` (embedding dimension) and
            ``pretrained`` (whether to use ImageNet weights).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        if config.pretrained:
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.resnet = models.resnet18(weights=None)

        # Modify first conv to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, config.d, bias=True)
        self.layer_norm = nn.LayerNorm(config.d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ECG signals.

        Args:
            x: ECG data ``(batch_sz, 1, 5000, 12)``.

        Returns:
            Learned ECG representation ``(batch_sz, d)``.
        """
        x = self.resnet(x)
        return self.layer_norm(x)


class LabsEncoder(nn.Module):
    """Laboratory results encoder using a 3-layer MLP.

    Encodes concatenated lab percentiles and missingness indicators
    into a normalized d-dimensional representation.

    Args:
        config: Configuration with ``d`` (embedding dimension).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, config.d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode laboratory results.

        Args:
            x: Concatenated lab percentiles and missingness ``(batch_sz, 100)``.

        Returns:
            Learned labs representation ``(batch_sz, d)``.
        """
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        return self.layer_norm(x)


class SymileMIMICModel(nn.Module):
    """3-modality contrastive learning model for SYMILE-MIMIC.

    Combines CXR, ECG, and Labs encoders with a learnable temperature
    parameter for contrastive loss computation.

    Args:
        config: Configuration with model architecture parameters:
            ``d``, ``pretrained``, ``freeze_logit_scale``, ``logit_scale_init``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        self.ecg_encoder = ECGEncoder(self.config)
        self.cxr_encoder = CXREncoder(self.config)
        self.labs_encoder = LabsEncoder(self.config)

        logit_scale = torch.ones([]) * self.config.logit_scale_init
        if self.config.freeze_logit_scale:
            self.logit_scale = nn.Parameter(logit_scale, requires_grad=False)
        else:
            self.logit_scale = nn.Parameter(logit_scale)

    def forward(
        self, x: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all three modalities.

        Args:
            x: List of [cxr, ecg, labs_percentiles, labs_missingness, hadm_id].

        Returns:
            Tuple of (r_c, r_e, r_l, logit_scale_exp) where each
            representation is ``(batch_sz, d)`` and logit_scale_exp is a scalar.
        """
        r_c = self.cxr_encoder(x[0])
        r_e = self.ecg_encoder(x[1])
        labs = torch.cat([x[2], x[3]], dim=1)
        r_l = self.labs_encoder(labs)

        return r_c, r_e, r_l, self.logit_scale.exp()

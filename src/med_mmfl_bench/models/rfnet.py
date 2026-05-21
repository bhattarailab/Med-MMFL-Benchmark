"""RFNet: Region-aware Fusion Network for incomplete multi-modal brain tumor segmentation.

Implements the RFNet architecture which handles missing MRI modalities
through region-aware modal fusion guided by preliminary region maps (PRMs).
Each modality (FLAIR, T1ce, T1, T2) is encoded independently, then
fused via learned per-region attention weights.

Reference:
    Ding et al., "RFNet: Region-aware Fusion Network for Incomplete
    Multi-modal Brain Tumor Segmentation",
    https://github.com/dyh127/RFNet

Architecture Overview::

    Input (B, 4, D, H, W)
    ├── FLAIR Encoder ──┐
    ├── T1ce  Encoder ──┤
    ├── T1    Encoder ──┤  ──►  Decoder_fuse (region-aware)  ──► fuse_pred
    └── T2    Encoder ──┘       ├── PRM generators at each scale
                                └── Region-aware modal fusion (RFM)

    During training, each modality is also independently decoded via
    Decoder_sep to produce per-modality auxiliary predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple

__all__ = ["RFNet"]

# Base feature map width used throughout the network.
BASIC_DIMS = 16


# ============================================================================
# Helper Modules
# ============================================================================


def normalization(planes: int, norm: str = "bn") -> nn.Module:
    """Create a normalization layer.

    Args:
        planes: Number of feature channels.
        norm: Normalization type. One of ``"bn"`` (BatchNorm),
            ``"gn"`` (GroupNorm), ``"in"`` (InstanceNorm).

    Returns:
        Normalization module.

    Raises:
        ValueError: If ``norm`` is not a supported type.
    """
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    elif norm == "gn":
        return nn.GroupNorm(4, planes)
    elif norm == "in":
        return nn.InstanceNorm3d(planes)
    else:
        raise ValueError(f"Normalization type '{norm}' is not supported.")


class GeneralConv3d(nn.Module):
    """General 3D convolution block: Conv3D → Norm → Activation.

    A reusable building block combining convolution, normalization, and
    activation into a single module.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        k_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        pad_type: Padding mode for ``nn.Conv3d`` (e.g., ``"reflect"``).
        norm: Normalization type (``"bn"``, ``"gn"``, ``"in"``).
        act_type: Activation type: ``"relu"`` or ``"lrelu"``.
        relu_factor: Negative slope for LeakyReLU.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "reflect",
        norm: str = "in",
        act_type: str = "lrelu",
        relu_factor: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(out_ch, norm=norm)

        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(
                negative_slope=relu_factor, inplace=True
            )
        else:
            raise ValueError(f"Activation type '{act_type}' is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv → norm → activation.

        Args:
            x: Input tensor ``(B, in_ch, D, H, W)``.

        Returns:
            Output tensor ``(B, out_ch, D', H', W')``.
        """
        return self.activation(self.norm(self.conv(x)))


# ============================================================================
# Encoder
# ============================================================================


class Encoder(nn.Module):
    """Single-modality 3D encoder with 4 resolution levels.

    Processes a single MRI modality through 4 stages of residual
    convolution blocks with progressive downsampling (stride 2).

    Each stage consists of three convolutions with a residual skip
    connection from the stage input to the output of the third conv.

    Output feature maps at each level:
        - Level 1: ``(B, 16, D, H, W)``
        - Level 2: ``(B, 32, D/2, H/2, W/2)``
        - Level 3: ``(B, 64, D/4, H/4, W/4)``
        - Level 4: ``(B, 128, D/8, H/8, W/8)``
    """

    def __init__(self) -> None:
        super().__init__()
        bd = BASIC_DIMS

        # Stage 1: 1 → 16 channels, full resolution
        self.e1_c1 = GeneralConv3d(1, bd, pad_type="reflect")
        self.e1_c2 = GeneralConv3d(bd, bd, pad_type="reflect")
        self.e1_c3 = GeneralConv3d(bd, bd, pad_type="reflect")

        # Stage 2: 16 → 32 channels, 1/2 resolution
        self.e2_c1 = GeneralConv3d(bd, bd * 2, stride=2, pad_type="reflect")
        self.e2_c2 = GeneralConv3d(bd * 2, bd * 2, pad_type="reflect")
        self.e2_c3 = GeneralConv3d(bd * 2, bd * 2, pad_type="reflect")

        # Stage 3: 32 → 64 channels, 1/4 resolution
        self.e3_c1 = GeneralConv3d(bd * 2, bd * 4, stride=2, pad_type="reflect")
        self.e3_c2 = GeneralConv3d(bd * 4, bd * 4, pad_type="reflect")
        self.e3_c3 = GeneralConv3d(bd * 4, bd * 4, pad_type="reflect")

        # Stage 4: 64 → 128 channels, 1/8 resolution
        self.e4_c1 = GeneralConv3d(bd * 4, bd * 8, stride=2, pad_type="reflect")
        self.e4_c2 = GeneralConv3d(bd * 8, bd * 8, pad_type="reflect")
        self.e4_c3 = GeneralConv3d(bd * 8, bd * 8, pad_type="reflect")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a single modality at 4 resolution levels.

        Args:
            x: Single-channel MRI volume ``(B, 1, D, H, W)``.

        Returns:
            Tuple of ``(x1, x2, x3, x4)`` feature maps at decreasing
            spatial resolutions.
        """
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4


# ============================================================================
# Decoders
# ============================================================================


class DecoderSep(nn.Module):
    """Per-modality decoder for auxiliary segmentation predictions.

    Standard U-Net-style decoder with skip connections. Produces a
    per-modality segmentation prediction used as an auxiliary training
    signal.

    Args:
        num_cls: Number of output segmentation classes.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        bd = BASIC_DIMS

        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = GeneralConv3d(bd * 8, bd * 4, pad_type="reflect")
        self.d3_c2 = GeneralConv3d(bd * 8, bd * 4, pad_type="reflect")
        self.d3_out = GeneralConv3d(bd * 4, bd * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = GeneralConv3d(bd * 4, bd * 2, pad_type="reflect")
        self.d2_c2 = GeneralConv3d(bd * 4, bd * 2, pad_type="reflect")
        self.d2_out = GeneralConv3d(bd * 2, bd * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = GeneralConv3d(bd * 2, bd, pad_type="reflect")
        self.d1_c2 = GeneralConv3d(bd * 2, bd, pad_type="reflect")
        self.d1_out = GeneralConv3d(bd, bd, k_size=1, padding=0, pad_type="reflect")

        self.seg_layer = nn.Conv3d(
            in_channels=bd, out_channels=num_cls,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> torch.Tensor:
        """Decode features into a segmentation prediction.

        Args:
            x1: Level-1 encoder features ``(B, 16, D, H, W)``.
            x2: Level-2 encoder features ``(B, 32, D/2, H/2, W/2)``.
            x3: Level-3 encoder features ``(B, 64, D/4, H/4, W/4)``.
            x4: Level-4 encoder features ``(B, 128, D/8, H/8, W/8)``.

        Returns:
            Softmax segmentation prediction ``(B, num_cls, D, H, W)``.
        """
        de_x4 = self.d3_c1(self.d3(x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        return self.softmax(logits)


class DecoderFuse(nn.Module):
    """Region-aware fusion decoder with PRM-guided modal attention.

    Generates Preliminary Region Maps (PRMs) at each decoder level, then
    uses them to guide modality-specific feature fusion via learned
    attention weights per semantic region.

    Args:
        num_cls: Number of output segmentation classes.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        bd = BASIC_DIMS

        # Decoder convolutions
        self.d3_c1 = GeneralConv3d(bd * 8, bd * 4, pad_type="reflect")
        self.d3_c2 = GeneralConv3d(bd * 8, bd * 4, pad_type="reflect")
        self.d3_out = GeneralConv3d(bd * 4, bd * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2_c1 = GeneralConv3d(bd * 4, bd * 2, pad_type="reflect")
        self.d2_c2 = GeneralConv3d(bd * 4, bd * 2, pad_type="reflect")
        self.d2_out = GeneralConv3d(bd * 2, bd * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1_c1 = GeneralConv3d(bd * 2, bd, pad_type="reflect")
        self.d1_c2 = GeneralConv3d(bd * 2, bd, pad_type="reflect")
        self.d1_out = GeneralConv3d(bd, bd, k_size=1, padding=0, pad_type="reflect")

        self.seg_layer = nn.Conv3d(
            in_channels=bd, out_channels=num_cls,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
        self.softmax = nn.Softmax(dim=1)

        # Upsampling layers for PRM predictions at different scales
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)

        # Region-aware modal fusion modules at each decoder level
        self.RFM4 = RegionAwareModalFusion(in_channel=bd * 8, num_cls=num_cls)
        self.RFM3 = RegionAwareModalFusion(in_channel=bd * 4, num_cls=num_cls)
        self.RFM2 = RegionAwareModalFusion(in_channel=bd * 2, num_cls=num_cls)
        self.RFM1 = RegionAwareModalFusion(in_channel=bd * 1, num_cls=num_cls)

        # Preliminary Region Map (PRM) generators
        self.prm_generator4 = PRMGeneratorLastStage(in_channel=bd * 8, num_cls=num_cls)
        self.prm_generator3 = PRMGenerator(in_channel=bd * 4, num_cls=num_cls)
        self.prm_generator2 = PRMGenerator(in_channel=bd * 2, num_cls=num_cls)
        self.prm_generator1 = PRMGenerator(in_channel=bd * 1, num_cls=num_cls)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Decode with region-aware fusion.

        Args:
            x1: Level-1 stacked modality features ``(B, K, C, D, H, W)``.
            x2: Level-2 stacked modality features.
            x3: Level-3 stacked modality features.
            x4: Level-4 stacked modality features.
            mask: Modality availability mask ``(B, K)`` where True
                indicates the modality is present.

        Returns:
            Tuple of:
                - ``pred``: Fused segmentation prediction ``(B, num_cls, D, H, W)``.
                - ``prm_preds``: Tuple of PRM predictions at each level,
                  all upsampled to full resolution.
        """
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        prm_preds = (
            prm_pred1,
            self.up2(prm_pred2),
            self.up4(prm_pred3),
            self.up8(prm_pred4),
        )
        return pred, prm_preds


# ============================================================================
# PRM Generators
# ============================================================================


class PRMGeneratorLastStage(nn.Module):
    """Preliminary Region Map generator for the deepest encoder stage.

    Generates a coarse region map from the bottleneck features using a
    bottleneck embedding followed by softmax classification.

    Args:
        in_channel: Number of input feature channels per modality.
        num_cls: Number of segmentation classes.
    """

    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(in_channel * 4, in_channel // 4, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1),
        )
        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate PRM from bottleneck features.

        Args:
            x: Stacked modality features ``(B, K, C, D, H, W)``.
            mask: Modality availability mask ``(B, K)``.

        Returns:
            Region probability map ``(B, num_cls, D, H, W)``.
        """
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)

        return self.prm_layer(self.embedding_layer(y))


class PRMGenerator(nn.Module):
    """Preliminary Region Map generator for intermediate decoder stages.

    Combines decoded features from the previous stage with encoder
    features at the current resolution to generate a region map.

    Args:
        in_channel: Number of input feature channels per modality.
        num_cls: Number of segmentation classes.
    """

    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(in_channel * 4, in_channel // 4, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1),
        )
        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel * 2, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate PRM from decoded and encoder features.

        Args:
            x1: Decoded features from previous stage ``(B, C, D, H, W)``.
            x2: Stacked encoder features ``(B, K, C, D, H, W)``.
            mask: Modality availability mask ``(B, K)``.

        Returns:
            Region probability map ``(B, num_cls, D, H, W)``.
        """
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        y = y.view(B, -1, H, W, Z)

        return self.prm_layer(torch.cat((x1, self.embedding_layer(y)), dim=1))


# ============================================================================
# Region-Aware Modal Fusion
# ============================================================================


class ModalFusion(nn.Module):
    """Per-region modality fusion with learned attention weights.

    For a given semantic region, computes attention weights over the K
    modalities based on region-averaged features, then produces a
    weighted sum of per-modality features.

    Args:
        in_channel: Number of feature channels per modality.
    """

    def __init__(self, in_channel: int = 64) -> None:
        super().__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv3d(4 * in_channel + 1, 128, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(128, 4, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        prm: torch.Tensor,
        region_name: str,
    ) -> torch.Tensor:
        """Fuse modality features for a specific region.

        Args:
            x: Per-modality region features ``(B, K, C, D, H, W)``.
            prm: Region probability map for this region ``(B, 1, C, D, H, W)``.
            region_name: Name of the region (for debugging; not used in
                computation).

        Returns:
            Fused feature map ``(B, C, D, H, W)``.
        """
        B, K, C, H, W, Z = x.size()

        prm_avg = torch.mean(prm, dim=(3, 4, 5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg

        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat(
            (feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1
        )
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        # Weighted sum across modalities
        return torch.sum(x * weight, dim=1)


class RegionFusion(nn.Module):
    """Fuse per-region features into a single feature map.

    Concatenates features from all regions and reduces the channel
    dimension through a bottleneck convolution sequence.

    Args:
        in_channel: Number of feature channels per region.
        num_cls: Number of regions (classes) to fuse.
    """

    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse region features.

        Args:
            x: Per-region features ``(B, num_cls, C, D, H, W)``.

        Returns:
            Fused feature map ``(B, C/2, D, H, W)``.
        """
        B, _, _, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)


class RegionAwareModalFusion(nn.Module):
    """Region-aware modal fusion module (RFM).

    Splits multi-modal features into per-region components using PRM
    guidance, fuses modalities within each region via learned attention,
    then combines region features with a skip connection from the raw
    concatenated modality features.

    Args:
        in_channel: Number of feature channels per modality.
        num_cls: Number of segmentation classes (regions).
    """

    def __init__(
        self, in_channel: int = 64, num_cls: int = 4
    ) -> None:
        super().__init__()
        self.num_cls = num_cls

        self.modal_fusion = nn.ModuleList([
            ModalFusion(in_channel=in_channel) for _ in range(num_cls)
        ])
        self.region_fusion = RegionFusion(
            in_channel=in_channel, num_cls=num_cls
        )
        self.short_cut = nn.Sequential(
            GeneralConv3d(in_channel * 4, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1),
        )

        self.clsname_list = ["BG", "NCR", "ED", "NET", "ET"]

    def forward(
        self,
        x: torch.Tensor,
        prm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multi-modal features with region-aware attention.

        Args:
            x: Stacked modality features ``(B, K, C, D, H, W)``.
            prm: Region probability map ``(B, num_cls, D, H, W)``.
            mask: Modality availability mask ``(B, K)``.

        Returns:
            Fused feature map ``(B, C, D, H, W)`` (channel dim is
            ``in_channel``, combining region fusion and shortcut outputs).
        """
        B, K, C, H, W, Z = x.size()

        # Mask unavailable modalities
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]

        # Expand PRM to match feature dims and split by region
        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)

        flair = y[:, 0:1, ...] * prm
        t1ce = y[:, 1:2, ...] * prm
        t1 = y[:, 2:3, ...] * prm
        t2 = y[:, 3:4, ...] * prm

        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        # Per-region modal fusion
        region_fused_feat = []
        for i in range(self.num_cls):
            fused = self.modal_fusion[i](
                region_feat[i], prm[:, i:i + 1, ...], self.clsname_list[i]
            )
            region_fused_feat.append(fused)
        region_fused_feat = torch.stack(region_fused_feat, dim=1)

        # Combine region fusion output with shortcut from raw features
        final_feat = torch.cat((
            self.region_fusion(region_fused_feat),
            self.short_cut(y.view(B, -1, H, W, Z)),
        ), dim=1)

        return final_feat


# ============================================================================
# Main Model
# ============================================================================


class RFNet(nn.Module):
    """RFNet: Region-aware Fusion Network for incomplete multi-modal segmentation.

    Uses four independent encoders (one per MRI modality: FLAIR, T1ce, T1, T2)
    and two decoders:
        - **Decoder_fuse**: Region-aware fusion decoder producing the primary
          segmentation via PRM-guided modality attention.
        - **Decoder_sep**: Per-modality auxiliary decoder used only during
          training for additional supervision.

    Args:
        config: Configuration object with ``n_classes`` specifying the number
            of segmentation classes.

    Attributes:
        flair_encoder: Encoder for FLAIR modality.
        t1ce_encoder: Encoder for T1ce modality.
        t1_encoder: Encoder for T1 modality.
        t2_encoder: Encoder for T2 modality.
        decoder_fuse: Region-aware fusion decoder.
        decoder_sep: Per-modality auxiliary decoder.
        is_training: Controls whether auxiliary outputs are produced.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        num_cls = config.n_classes
        self.decoder_fuse = DecoderFuse(num_cls=num_cls)
        self.decoder_sep = DecoderSep(num_cls=num_cls)

        self.is_training = False

        # Kaiming initialization for all Conv3d layers
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> "torch.Tensor | Tuple":
        """Forward pass through RFNet.

        Args:
            x: Multi-modal input volume ``(B, 4, D, H, W)`` with channels
                ordered as [FLAIR, T1ce, T1, T2].
            mask: Modality availability mask ``(B, 4)`` where True indicates
                the modality is present.

        Returns:
            - **Inference mode** (``is_training=False``):
              Fused segmentation prediction ``(B, num_cls, D, H, W)``.

            - **Training mode** (``is_training=True``):
              Tuple of:
                - ``fuse_pred``: Fused prediction ``(B, num_cls, D, H, W)``.
                - ``bottleneck_feats``: Tuple of 4 bottleneck features,
                  one per modality ``(B, 128, D/8, H/8, W/8)``.
                - ``sep_preds``: Tuple of 4 per-modality predictions.
                - ``prm_preds``: Tuple of 4 PRM predictions at full resolution.
        """
        # Encode each modality independently
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        # Stack modalities: (B, 4, C, D, H, W) at each level
        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1)
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)

        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)
            return (
                fuse_pred,
                (flair_x4, t1ce_x4, t1_x4, t2_x4),
                (flair_pred, t1ce_pred, t1_pred, t2_pred),
                prm_preds,
            )

        return fuse_pred
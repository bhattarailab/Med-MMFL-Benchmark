"""BraTS 2024 GLI dataset for 3D brain tumor segmentation.

Loads multi-modal NIfTI brain MRI volumes and segmentation masks,
with support for patch extraction and preprocessing.

Requires: ``nibabel``
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError:
    raise ImportError(
        "nibabel is required for BraTS datasets. "
        "Install it with: pip install nibabel"
    )

__all__ = ["BraTS24GLIPostDataset"]


class BraTS24GLIPostDataset(Dataset):
    """BraTS 2024 GLI post-processed dataset for 3D segmentation.

    Loads multi-modal brain MRI volumes (T1c, T1n, T2w, T2f) and their
    corresponding segmentation masks. Supports tumor-centered patch
    extraction for efficient training.

    Args:
        data_dir: Root directory containing the NIfTI image files.
        ann_root: Directory containing the annotation pickle file.
        view_type: View identifier for selecting the annotation split file.
        split: Dataset split: ``"train"``, ``"val"``, or ``"test"``.
        modalities: List of MRI modalities to load.
        apply_augmentation: Whether to apply data augmentation (reserved
            for future use).
        patch_size: 3D patch dimensions ``(D, H, W)`` for extraction.

    Note:
        This dataset implementation is currently experimental. The BraTS
        training pipeline is under active development.
    """

    _MODALITY_INDICES = {"t1c": 0, "t1n": 1, "t2w": 2, "t2f": 3}

    def __init__(
        self,
        data_dir: str,
        ann_root: Optional[str] = None,
        view_type: str = "view1",
        split: str = "train",
        modalities: Optional[List[str]] = None,
        apply_augmentation: bool = False,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        import pickle

        if modalities is None:
            modalities = ["t1c", "t1n", "t2w", "t2f"]

        annotation_file = f"{ann_root}/brats24-{view_type}.pkl"
        with open(annotation_file, "rb") as f:
            loaded_data = pickle.load(f)

        self.data = loaded_data[split]
        self.image_root = data_dir
        self.modalities = modalities
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def load_nifti_image(path: str) -> np.ndarray:
        """Load a NIfTI image file as a NumPy array.

        Args:
            path: Path to the ``.nii`` or ``.nii.gz`` file.

        Returns:
            Image data as a float32 NumPy array.
        """
        return nib.load(path).get_fdata().astype(np.float32)

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Min-max normalize an array to [0, 1].

        Args:
            data: Input array.

        Returns:
            Normalized array. Returns zeros if max == min.
        """
        data_min = np.min(data)
        data_range = np.max(data) - data_min
        if data_range == 0:
            return np.zeros_like(data)
        return (data - data_min) / data_range

    def preprocess_modalities(self, image: np.ndarray) -> np.ndarray:
        """Normalize and reorder axes for a modality volume.

        Args:
            image: Raw 3D volume ``(X, Y, Z)``.

        Returns:
            Preprocessed volume ``(Z, Y, X)`` normalized to [0, 1].
        """
        image = self.normalize(image)
        return np.moveaxis(image, (0, 1, 2), (2, 1, 0))

    @staticmethod
    def preprocess_mask_labels(mask: np.ndarray) -> np.ndarray:
        """
        Convert a BraTS segmentation mask into 4-channel one-hot encoded binary masks.

        The input mask uses the following label convention:
            0 = Background (BG)
            1 = Non-Enhancing Tumor (NET)
            2 = Edema (ED)
            3 = Enhancing Tumor (ET)
            4 = Necrosis (merged into Background)

        Each label is separated into its own binary channel. Label 4 (necrosis)
        is treated as background and merged into channel 0.

        Args:
            mask (np.ndarray): Integer segmentation mask of shape (H, W) or (D, H, W),
                            with values in {0, 1, 2, 3, 4}.

        Returns:
            np.ndarray: One-hot encoded mask of shape (4, H, W) or (4, D, H, W),
                        with channels ordered as [BG, NET, ED, ET].
                        Each channel contains binary values {0, 1}.

        Channel mapping:
            Channel 0 (BG)  — 1 where mask ∈ {0, 4}, else 0
            Channel 1 (NET) — 1 where mask == 1, else 0
            Channel 2 (ED)  — 1 where mask == 2, else 0
            Channel 3 (ET)  — 1 where mask == 3, else 0
        """

        mask_BG = mask.copy()
        mask_BG[mask == 0] = 1
        mask_BG[mask == 1] = 0
        mask_BG[mask == 2] = 0
        mask_BG[mask == 3] = 0
        mask_BG[mask == 4] = 1
        
        mask_NET = mask.copy()
        mask_NET[mask == 0] = 0
        mask_NET[mask == 1] = 1
        mask_NET[mask == 2] = 0
        mask_NET[mask == 3] = 0
        mask_NET[mask == 4] = 0 

        mask_ED = mask.copy()
        mask_ED[mask == 0] = 0
        mask_ED[mask == 1] = 0
        mask_ED[mask == 2] = 1
        mask_ED[mask == 3] = 0
        mask_ED[mask == 4] = 0
        
        mask_ET = mask.copy()
        mask_ET[mask == 0] = 0
        mask_ET[mask == 1] = 0
        mask_ET[mask == 2] = 0
        mask_ET[mask == 3] = 1
        mask_ET[mask == 4] = 0
        
        mask = np.stack([mask_BG, mask_NET, mask_ED, mask_ET])
        return mask

        # mask_wt = np.isin(mask, [1, 2, 3]).astype(np.float32)
        # mask_tc = np.isin(mask, [1, 3]).astype(np.float32)
        # mask_et = (mask == 3).astype(np.float32)
        # return np.stack([mask_wt, mask_tc, mask_et])

    def extract_patch(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        oh_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a 3D patch centered on tumor core when possible.

        If tumor core voxels exist, the patch is centered on a randomly
        selected TC voxel. Otherwise, a random patch is extracted.

        Args:
            image: Full volume ``(C, D, H, W)``.
            label: Full label volume ``(1, D, H, W)``.
            oh_label: One-hot label ``(3, D, H, W)``.

        Returns:
            Tuple of (image_patch, label_patch, oh_label_patch) each
            cropped to ``self.patch_size``.
        """
        _, depth, height, width = label.shape
        pd, ph, pw = self.patch_size

        tc_mask = oh_label[1]  # TC channel
        coords = torch.nonzero(tc_mask)

        if len(coords) == 0:
            center_z = random.randint(pd // 2, depth - pd // 2)
            center_y = random.randint(ph // 2, height - ph // 2)
            center_x = random.randint(pw // 2, width - pw // 2)
        else:
            z, y, x = coords[random.randint(0, len(coords) - 1)]
            center_z = int(torch.clamp(z, pd // 2, depth - pd // 2))
            center_y = int(torch.clamp(y, ph // 2, height - ph // 2))
            center_x = int(torch.clamp(x, pw // 2, width - pw // 2))

        sz, ez = center_z - pd // 2, center_z + pd // 2
        sy, ey = center_y - ph // 2, center_y + ph // 2
        sx, ex = center_x - pw // 2, center_x + pw // 2

        return (
            image[:, sz:ez, sy:ey, sx:ex],
            label[:, sz:ez, sy:ey, sx:ex],
            oh_label[:, sz:ez, sy:ey, sx:ex],
        )

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample with patch extraction.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (images, label, oh_label) where images is ``(C, D, H, W)``,
            label is ``(D, H, W)``, and oh_label is ``(3, D, H, W)``.
        """
        data_item = self.data[idx]
        image_paths = [
            os.path.join(self.image_root, data_item[mod])
            for mod in self.modalities
        ]
        label_path = os.path.join(self.image_root, data_item["seg_path"])

        images = [
            self.preprocess_modalities(self.load_nifti_image(p))
            for p in image_paths
        ]
        label = self.load_nifti_image(label_path)
        label = np.moveaxis(label, (0, 1, 2), (2, 1, 0))
        images_np = np.stack(images, axis=0)

        oh_label = self.preprocess_mask_labels(label)

        images_t = torch.from_numpy(images_np).float()
        label_t = torch.from_numpy(label).long()
        oh_label_t = torch.from_numpy(oh_label).float()

        images_t, label_t, oh_label_t = self.extract_patch(
            images_t, label_t.unsqueeze(0), oh_label_t
        )

        # present modalities mask (for RFNet Model Architecture), keep all for now -> no absent modalities
        mask = torch.tensor([True, True, True, True])

        # return the idx of the batch too (not received by the model, but useful for evaluation and debugging)
        return images_t, label_t.squeeze(0), oh_label_t, mask, idx 

"""SYMILE-MIMIC dataset classes for 3-modality contrastive learning.

Loads pre-processed NumPy arrays of CXR images, ECG signals, and lab
results from the SYMILE-MIMIC dataset for federated multimodal training
and zero-shot retrieval evaluation.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["SymileMIMICDataset", "SymileMIMICRetrievalDataset"]


class SymileMIMICDataset(Dataset):
    """Training/validation dataset for SYMILE-MIMIC 3-modality learning.

    Loads memory-mapped NumPy arrays for efficient data access without
    loading the full dataset into memory.

    Args:
        data_dir: Root directory containing split subdirectories with
            ``.npy`` files (e.g., ``data_dir/train/cxr_train.npy``).
        ann_root: Annotation root (unused, kept for API compatibility).
        view_type: View type identifier (unused, kept for API compatibility).
        split: Dataset split: ``"train"``, ``"val"``, or ``"test"``.
    """

    def __init__(
        self,
        data_dir: str,
        ann_root: Any = None,
        view_type: str = "view1",
        split: str = "train",
    ) -> None:
        self.data_dir = data_dir
        self.split = split

        split_dir = os.path.join(self.data_dir, split)
        self.cxr = np.load(
            os.path.join(split_dir, f"cxr_{split}.npy"), mmap_mode="r"
        )
        self.ecg = np.load(
            os.path.join(split_dir, f"ecg_{split}.npy"), mmap_mode="r"
        )
        self.labs_percentiles = np.load(
            os.path.join(split_dir, f"labs_percentiles_{split}.npy"),
            mmap_mode="r",
        )
        self.labs_missingness = np.load(
            os.path.join(split_dir, f"labs_missingness_{split}.npy"),
            mmap_mode="r",
        )
        self.hadm_ids = np.load(
            os.path.join(split_dir, f"hadm_id_{split}.npy"), mmap_mode="r"
        )

    def __len__(self) -> int:
        return len(self.hadm_ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (cxr, ecg, labs_percentiles, labs_missingness, hadm_id, idx).
        """
        return (
            torch.tensor(self.cxr[idx]),
            torch.tensor(self.ecg[idx]),
            torch.tensor(self.labs_percentiles[idx]),
            torch.tensor(self.labs_missingness[idx]),
            torch.tensor(self.hadm_ids[idx]),
            idx,
        )


class SymileMIMICRetrievalDataset(Dataset):
    """Retrieval evaluation dataset for SYMILE-MIMIC.

    Loads the full dataset eagerly into memory (not memory-mapped) for
    efficient retrieval evaluation. Includes labels for positive/negative
    candidate identification.

    Args:
        data_dir: Root directory containing split subdirectories.
        split: Dataset split: ``"val_retrieval"`` or ``"test"``.
    """

    def __init__(self, data_dir: str, split: str) -> None:
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.cxr = torch.tensor(
            np.load(os.path.join(split_dir, f"cxr_{split}.npy"))
        )
        self.ecg = torch.tensor(
            np.load(os.path.join(split_dir, f"ecg_{split}.npy"))
        )
        self.labs_percentiles = torch.tensor(
            np.load(os.path.join(split_dir, f"labs_percentiles_{split}.npy"))
        )
        self.labs_missingness = torch.tensor(
            np.load(os.path.join(split_dir, f"labs_missingness_{split}.npy"))
        )
        self.hadm_id = torch.tensor(
            np.load(os.path.join(split_dir, f"hadm_id_{split}.npy"))
        )
        self.label_hadm_id = torch.tensor(
            np.load(os.path.join(split_dir, f"label_hadm_id_{split}.npy"))
        )
        self.label = torch.tensor(
            np.load(os.path.join(split_dir, f"label_{split}.npy"))
        )

    def __len__(self) -> int:
        return len(self.ecg)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single retrieval sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys: ``idx``, ``cxr``, ``ecg``,
            ``labs_percentiles``, ``labs_missingness``, ``hadm_id``,
            ``label_hadm_id``, ``label``.
        """
        return {
            "idx": idx,
            "cxr": self.cxr[idx],
            "ecg": self.ecg[idx],
            "labs_percentiles": self.labs_percentiles[idx],
            "labs_missingness": self.labs_missingness[idx],
            "hadm_id": self.hadm_id[idx],
            "label_hadm_id": self.label_hadm_id[idx],
            "label": self.label[idx],
        }

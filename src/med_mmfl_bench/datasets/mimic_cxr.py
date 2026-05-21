"""MIMIC-CXR multimodal dataset for image-text classification.

Loads chest X-ray images and radiology reports from the MIMIC-CXR
dataset for multi-label classification tasks.
"""

import pickle
import re
from typing import Any, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ["MimicMultiModal", "MimicPublic", "clean_report_mimic_cxr"]


def clean_report_mimic_cxr(report: str) -> str:
    """Clean and normalize a MIMIC-CXR radiology report.

    Removes numbered lists, excessive whitespace, repeated punctuation,
    and special characters. Converts to lowercase.

    Args:
        report: Raw radiology report text.

    Returns:
        Cleaned report string with sentences separated by `` . ``.
    """
    # Normalize whitespace and remove carriage returns
    report = re.sub(r"\s+", " ", report.replace("\r", ""))
    # Remove numbered list prefixes
    report = re.sub(r"\.\s*\d+\.\s*", ". ", report)
    report = re.sub(r"\s+\d+\.\s*", ". ", report)
    # Remove repeated dots
    report = re.sub(r"\.{2,}", ".", report)
    # Strip and lowercase
    report = report.strip().lower()
    # Split into sentences
    sentences = report.split(". ")
    # Clean each sentence
    sent_cleaner = re.compile(r'[.,?;*!%^&_+():\-\[\]{}]')
    tokens = [
        sent_cleaner.sub("", sent.replace('"', "").replace("/", "")
                         .replace("\\", "").replace("'", "").strip())
        for sent in sentences
    ]
    tokens = [t for t in tokens if t]
    return " . ".join(tokens) + " ."


def _get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """Get image transforms for the given split.

    Args:
        split: Dataset split (``"train"`` or validation/test).
        img_size: Target image size.

    Returns:
        Composed transform pipeline.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


class MimicMultiModal(Dataset):
    """MIMIC-CXR multimodal dataset with images and reports.

    Loads chest X-ray images and their associated radiology reports
    for multi-label pathology classification.

    Args:
        data_root: Root directory containing the CXR image files.
        ann_root: Directory containing the annotation pickle file.
        view_type: View type filter (e.g., ``"view1"``, ``"APPA"``).
        split: Dataset split: ``"train"``, ``"val"``, or ``"test"``.
        img_size: Target image size for resizing.
    """

    def __init__(
        self,
        data_root: str,
        ann_root: str,
        view_type: str = "view1",
        split: str = "train",
        img_size: int = 224,
    ) -> None:
        super().__init__()

        ann_file = f"{ann_root}/mimic-cxr-{view_type}.pkl"
        with open(ann_file, "rb") as f:
            loaded_data = pickle.load(f)

        self.data = loaded_data[split]
        self.transforms = _get_transforms(split, img_size)
        self.image_root = data_root

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_tensor, cleaned_report, index).
        """
        data_item = self.data[idx]
        report = data_item["report"]

        relative_path = data_item["image_path"][0].replace("jpg", "png")
        img_path = self.image_root + relative_path
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        label = data_item["label"]
        cleaned_report = clean_report_mimic_cxr(report)
        return image, torch.tensor(label), cleaned_report, idx


class MimicPublic(Dataset):
    """MIMIC-CXR public dataset variant (training split only).

    Similar to ``MimicMultiModal`` but always loads from the training split,
    suitable for public data distillation scenarios.

    Args:
        data_root: Root directory containing image files.
        ann_root: Directory containing the annotation pickle file.
        view_type: View type filter.
        dst_type: Transform type: ``"train"`` for augmented, else standard.
        img_size: Target image size.
    """

    def __init__(
        self,
        data_root: str,
        ann_root: str,
        view_type: str = "view1",
        dst_type: str = "train",
        img_size: int = 224,
    ) -> None:
        super().__init__()

        ann_file = f"{ann_root}/mimic-cxr-{view_type}.pkl"
        with open(ann_file, "rb") as f:
            loaded_data = pickle.load(f)

        self.data = loaded_data["train"]
        self.transforms = _get_transforms(dst_type, img_size)
        self.image_root = data_root

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_tensor, cleaned_report, index).
        """
        data_item = self.data[idx]
        report = data_item["report"]

        relative_path = data_item["image_path"][0].replace("jpg", "png")
        img_path = self.image_root + relative_path
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        label = data_item["label"]
        cleaned_report = clean_report_mimic_cxr(report)
        return image, torch.tensor(label), cleaned_report, idx

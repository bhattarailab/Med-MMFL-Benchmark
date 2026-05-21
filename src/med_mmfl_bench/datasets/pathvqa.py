"""PathVQA Yes/No visual question answering dataset.

Loads pathology images and binary (Yes/No) questions from a
pipe-delimited text file for classification-based VQA using
BLIP-style models.

Requires: ``transformers``
"""

import os
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BlipImageProcessor, BlipProcessor

from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["YesNoVQADataset"]

# Supported image extensions for fallback resolution
_IMAGE_EXTENSIONS = (".png", ".jpeg", ".JPG", ".PNG", ".JPEG")


class YesNoVQADataset(Dataset):
    """Binary (Yes/No) visual question answering dataset for pathology images.

    Loads data from a pipe-delimited text file where each line has the
    format: ``image_id|question|answer``. Answers are mapped to binary
    labels (``yes`` → 1, ``no`` → 0).

    Args:
        data_dir: Directory containing pathology image files.
        ann_root: Directory containing ``{split}_yn.txt`` annotation files.
        view: View identifier (unused, kept for API compatibility).
        split: Dataset split: ``"train"``, ``"val"``, or ``"test"``.
        *args: Additional positional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Attributes:
        data: List of parsed sample dictionaries with ``image_id``,
            ``question``, and ``label`` keys.
    """

    def __init__(
        self,
        data_dir: str,
        ann_root: str,
        view: str,
        split: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.image_dir = data_dir
        self.text_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        self.image_processor = BlipImageProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        self.max_length = 32
        self.image_height = 384
        self.image_width = 384
        self.split = split

        txt_file_path = os.path.join(ann_root, f"{split}_yn.txt")
        self.data: List[Dict[str, Any]] = []

        with open(txt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) == 3:
                    image_id, question, answer = parts
                    label = 1 if answer.lower().strip() == "yes" else 0
                    self.data.append({
                        "image_id": image_id,
                        "question": question,
                        "label": label,
                    })

    def _resolve_image_path(self, image_id: str) -> str:
        """Resolve the full image path, trying multiple extensions.

        Args:
            image_id: Base image identifier (without extension).

        Returns:
            Resolved image path. Falls back to ``.jpg`` if no match found.
        """
        path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if os.path.exists(path):
            return path

        for ext in _IMAGE_EXTENSIONS:
            candidate = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                return candidate

        return path  # Return default .jpg path (will fail gracefully below)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single VQA sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys: ``input_ids``, ``attention_mask``,
            ``pixel_values``, ``labels``. All tensors have the batch
            dimension squeezed.
        """
        item = self.data[idx]

        image_path = self._resolve_image_path(item["image_id"])
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            logger.debug("Image not found or unreadable: %s", image_path)
            image = Image.new("RGB", (224, 224), color="white")

        text = item["question"]

        image_encoding = self.image_processor(
            image,
            do_resize=True,
            size=(self.image_height, self.image_width),
            return_tensors="pt",
        )

        encoding = self.text_processor(
            None,
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension added by the processor
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        encoding["pixel_values"] = image_encoding["pixel_values"][0]

        encoding["labels"] = torch.tensor(item["label"], dtype=torch.long)

        return encoding
"""EHRXQA dataset for EHR-grounded visual question answering.

Loads chest X-ray images and structured EHR context (clinical records,
lab results, etc.) paired with natural-language questions and answers
for generative VQA tasks using BLIP-style models.

Requires: ``transformers``
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BlipImageProcessor, BlipProcessor

from med_mmfl_bench.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["EHRXQA"]


class EHRXQA(Dataset):
    """EHR-grounded X-ray Question Answering dataset.

    Each sample consists of a clinical question, structured EHR context
    (lab results, procedures, etc.), a chest X-ray image, and a
    free-text answer. The text and image are pre-processed using the
    BLIP processor for VQA model consumption.

    Args:
        root_img_path: Root directory containing CXR image files.
        json_path: Directory containing ``{split}_questions.json`` files.
        view: View identifier (unused, kept for API compatibility).
        split: Dataset split: ``"train"``, ``"val"``, or ``"test"``.
        max_text_len: Maximum token length for the question + context input.
        max_ans_len: Maximum token length for the answer.

    Attributes:
        data: List of question dictionaries loaded from JSON.
        text_processor: BLIP text processor for tokenization.
        image_processor: BLIP image processor for resizing/normalization.
    """

    def __init__(
        self,
        root_img_path: str,
        json_path: str,
        view: str,
        split: str,
        max_text_len: int = 32,
        max_ans_len: int = 128,
    ) -> None:
        self.root_img_path = root_img_path
        self.max_text_len = max_text_len
        self.max_ans_len = max_ans_len

        questions_file = os.path.join(json_path, f"{split}_questions.json")
        with open(questions_file, "r") as f:
            self.data = json.load(f)

        self.text_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        self.image_processor = BlipImageProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        self.max_length = max_text_len
        self.image_height = 384
        self.image_width = 384

    def _process_answer(self, answer: Any) -> str:
        """Normalize an answer value to a non-empty string.

        Args:
            answer: Raw answer value (str, list, or None).

        Returns:
            Cleaned answer string, or ``"No answer"`` if empty/None.
        """
        if answer is None:
            return "No answer"

        if isinstance(answer, list):
            answer = " ".join(str(a) for a in answer)
        else:
            answer = str(answer)

        if len(answer.strip()) == 0:
            return "No answer"
        return answer

    @staticmethod
    def _build_context_text(context: List[Dict[str, Any]]) -> str:
        """Format structured EHR context into a text block.

        Args:
            context: List of context dictionaries, each mapping a record
                type name to a list of records.

        Returns:
            Formatted context string.
        """
        text_context = "Context:\nCurrent Time: 2105-12-31 23:59:00\n"
        for context_item in context:
            for record_name, records in context_item.items():
                text_context += f"{record_name}:\n"
                for i, rec in enumerate(records):
                    text_context += f"Record {i}:\n"
                    if isinstance(rec, str):
                        text_context += rec + "\n"
                        continue
                    for k, v in rec.items():
                        text_context += f"{k}: {v}\n"
                    text_context += "\n"
        return text_context

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single EHRXQA sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys: ``input_ids``, ``attention_mask``,
            ``pixel_values``, ``labels``. All tensors have the batch
            dimension squeezed.
        """
        item = self.data[idx]
        question = str(item.get("question", ""))
        context = item.get("context", [])

        text_context = self._build_context_text(context)
        text_input = f"Question: {question}\n\n{text_context}\n"

        # Load image (fallback to white placeholder if missing)
        images_path = item.get("images_path", None)
        if images_path and self.root_img_path:
            image_file = os.path.join(
                self.root_img_path, random.choice(images_path)
            )
            if os.path.exists(image_file):
                image = Image.open(image_file).convert("RGB")
            else:
                logger.debug("Image not found: %s", image_file)
                image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        else:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        image_encoding = self.image_processor(
            images=image,
            do_resize=True,
            size=(self.image_height, self.image_width),
            return_tensors="pt",
        )

        encoding = self.text_processor(
            images=None,
            text=text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        answer = self._process_answer(item.get("answer", None))
        labels = self.text_processor.tokenizer.encode(
            answer,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        encoding["labels"] = labels

        # Remove batch dimension added by the processor
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        encoding["pixel_values"] = image_encoding["pixel_values"][0]

        return encoding
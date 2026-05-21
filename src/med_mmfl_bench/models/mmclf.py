"""Multimodal classifier models for MIMIC-CXR image-text classification.

Provides ResNet-based image encoders, BERT-based text encoders, and
combined multimodal/split classifiers for 14-label pathology classification.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = [
    "MultiModalClassifier",
    "TextSplitClassifier",
    "ImageSplitClassifier",
    "EncoderBert",
    "EncoderResNet",
]


def l2_normalize(tensor: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """L2-normalize columns of a tensor.

    Args:
        tensor: Input tensor.
        axis: Dimension along which to normalize.

    Returns:
        L2-normalized tensor.
    """
    return F.normalize(tensor, p=2, dim=axis)


class EncoderBert(nn.Module):
    """BERT-based text encoder for radiology reports.

    Args:
        embed_dim: Output embedding dimension.
        txt_type: BERT variant: ``"bert-base-uncased"`` or ``"tinyBert"``.
    """

    def __init__(self, embed_dim: int, txt_type: str) -> None:
        super().__init__()
        from transformers import BertModel

        if txt_type == "bert-base-uncased":
            self.txt_enc = BertModel.from_pretrained("bert-base-uncased")
            self.linear = nn.Linear(768, embed_dim)
        elif txt_type == "tinyBert":
            self.txt_enc = BertModel.from_pretrained("huawei-noah/TinyBERT_4L_zh")
            self.linear = nn.Linear(312, embed_dim)
        else:
            raise ValueError(f"Unsupported txt_type: '{txt_type}'")

    def forward(
        self, tokenizer: Any, sentences: Any
    ) -> Dict[str, torch.Tensor]:
        """Encode text sentences.

        Args:
            tokenizer: BERT tokenizer instance.
            sentences: List of text strings or batch of sentences.

        Returns:
            Dictionary with ``"embedding"`` key containing L2-normalized
            text embeddings ``(batch_sz, embed_dim)``.
        """
        inputs = tokenizer(
            sentences,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.txt_enc(**inputs)
        embedding = l2_normalize(self.linear(out["last_hidden_state"][:, 0, :]))
        return {"embedding": embedding}


class EncoderResNet(nn.Module):
    """ResNet-based image encoder for chest X-rays.

    Args:
        embed_dim: Output embedding dimension.
        cnn_type: ResNet variant name (e.g., ``"resnet50"``).
    """

    def __init__(self, embed_dim: int, cnn_type: str) -> None:
        super().__init__()
        self.cnn = getattr(models, cnn_type)(weights="DEFAULT")
        self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()
        self.fc = nn.Linear(self.cnn_dim, embed_dim)
        self.cnn.fc = nn.Sequential()

    def init_weights(self) -> None:
        """Initialize FC layer weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images.

        Args:
            images: Input images ``(batch_sz, 3, H, W)``.

        Returns:
            Dictionary with ``"embedding"`` key containing L2-normalized
            image embeddings ``(batch_sz, embed_dim)``.
        """
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = l2_normalize(self.fc(pooled))
        return {"embedding": out}


class MultiModalClassifier(nn.Module):
    """Combined image-text classifier for 14-label pathology detection.

    Args:
        config: Configuration with ``embed_dim``, ``cnn_type``, and ``txt_type``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.image_encoder = EncoderResNet(
            embed_dim=config.embed_dim, cnn_type=config.cnn_type
        )
        self.text_encoder = EncoderBert(
            config.embed_dim, txt_type=config.txt_type
        )
        self.fc = nn.Linear(2 * config.embed_dim, 14)

    def forward(
        self, tokenizer: Any, img: torch.Tensor, txt: Any
    ) -> Dict[str, torch.Tensor]:
        """Forward pass combining image and text features.

        Args:
            tokenizer: BERT tokenizer for text encoding.
            img: Input images ``(batch_sz, 3, H, W)``.
            txt: List of radiology report strings.

        Returns:
            Dictionary with ``"logits"``, ``"image_features"``, and
            ``"caption_features"`` keys.
        """
        embed_img = self.image_encoder(img)
        embed_text = self.text_encoder(tokenizer, txt)
        concat_embed = torch.cat(
            (embed_img["embedding"], embed_text["embedding"]), dim=1
        )
        out = self.fc(concat_embed)
        return {
            "logits": out,
            "image_features": embed_img["embedding"],
            "caption_features": embed_text["embedding"],
        }


class TextSplitClassifier(nn.Module):
    """Text-only classifier with zero-padded image features.

    Used for evaluating unimodal text performance in the multimodal
    classification framework.

    Args:
        config: Configuration with ``embed_dim`` and ``txt_type``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.text_encoder = EncoderBert(config.embed_dim, txt_type=config.txt_type)
        self.fc = nn.Linear(2 * config.embed_dim, 14)

    def forward(
        self, tokenizer: Any, txt: Any
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with text only.

        Args:
            tokenizer: BERT tokenizer.
            txt: List of text strings.

        Returns:
            Dictionary with ``"logits"``, ``"image_features"`` (zeros),
            and ``"caption_features"``.
        """
        embed_text = self.text_encoder(tokenizer, txt)
        embed_img = torch.zeros_like(embed_text["embedding"])
        concat_embed = torch.cat((embed_img, embed_text["embedding"]), dim=1)
        return {
            "logits": self.fc(concat_embed),
            "image_features": embed_img,
            "caption_features": embed_text["embedding"],
        }


class ImageSplitClassifier(nn.Module):
    """Image-only classifier with zero-padded text features.

    Used for evaluating unimodal image performance in the multimodal
    classification framework.

    Args:
        config: Configuration with ``embed_dim`` and ``cnn_type``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.image_encoder = EncoderResNet(
            embed_dim=config.embed_dim, cnn_type=config.cnn_type
        )
        self.fc = nn.Linear(2 * config.embed_dim, 14)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with images only.

        Args:
            img: Input images ``(batch_sz, 3, H, W)``.

        Returns:
            Dictionary with ``"logits"``, ``"image_features"``, and
            ``"caption_features"`` (zeros).
        """
        embed_img = self.image_encoder(img)
        embed_text = torch.zeros_like(embed_img["embedding"])
        concat_embed = torch.cat((embed_img["embedding"], embed_text), dim=1)
        return {
            "logits": self.fc(concat_embed),
            "image_features": embed_img["embedding"],
            "caption_features": embed_text,
        }

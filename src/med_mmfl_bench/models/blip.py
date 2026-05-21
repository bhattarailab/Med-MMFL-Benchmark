"""BLIP-based models for visual question answering tasks.

Provides two model variants built on top of ``BlipForQuestionAnswering``:
    - :class:`BlipForEHRXQA`: Generative VQA for EHR-grounded questions.
    - :class:`BlipForYesNoVQA`: Binary classification VQA for Yes/No questions.

Reference:
    Li et al., "BLIP: Bootstrapping Language-Image Pre-training for Unified
    Vision-Language Understanding and Generation", ICML 2022.
    https://arxiv.org/abs/2201.12086

Requires: ``transformers``
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BlipForQuestionAnswering, BlipProcessor

__all__ = ["BlipForEHRXQA", "BlipForYesNoVQA"]


class BlipForEHRXQA(nn.Module):
    """BLIP model for EHR-grounded generative visual question answering.

    Wraps ``BlipForQuestionAnswering`` with an optional CLS-token
    representation output for knowledge distillation in federated
    learning settings (CreamFL, etc.).

    Args:
        config: Configuration object with ``pretrained_model_name``
            specifying the HuggingFace model identifier.

    Attributes:
        blip: Underlying BLIP VQA model.
        tokenizer: BLIP tokenizer for answer decoding.
        return_repr: If True, ``forward()`` returns CLS representations
            alongside model outputs.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.return_repr = False
        self.blip = BlipForQuestionAnswering.from_pretrained(
            config.pretrained_model_name
        )
        self.tokenizer = BlipProcessor.from_pretrained(
            config.pretrained_model_name
        ).tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """Forward pass for generative VQA.

        Args:
            input_ids: Tokenized question ``(B, seq_len)``.
            attention_mask: Attention mask ``(B, seq_len)``.
            pixel_values: Preprocessed images ``(B, 3, H, W)``.
            labels: Target answer token IDs ``(B, ans_len)``, or None
                for inference.

        Returns:
            - If ``return_repr`` is False: BLIP model outputs (loss, logits).
            - If ``return_repr`` is True: Tuple of ``(cls_repr, outputs)``
              where ``cls_repr`` is the CLS-token embedding ``(B, D)``.
        """
        # Extract CLS representation via the text encoder
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values)
        question_outputs = self.blip.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_outputs.last_hidden_state,
            return_dict=True,
        )
        text_repr = question_outputs.last_hidden_state[:, 0]   # (B, D)
        vision_repr = vision_outputs.last_hidden_state[:, 0]   # (B, D)

        # Full BLIP forward pass (includes answer generation / loss)
        outputs = self.blip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
        )

        if self.return_repr:
            return text_repr, vision_repr , outputs
        
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_length: int = 50,
    ) -> torch.Tensor:
        """Generate answer tokens for a given question and image.

        Args:
            input_ids: Tokenized question ``(B, seq_len)``.
            attention_mask: Attention mask ``(B, seq_len)``.
            pixel_values: Preprocessed images ``(B, 3, H, W)``.
            max_length: Maximum number of tokens to generate.

        Returns:
            Generated token IDs ``(B, generated_len)``.
        """
        return self.blip.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_length=max_length,
        )


class BlipForYesNoVQA(nn.Module):
    """BLIP model for binary (Yes/No) visual question answering.

    Uses the BLIP vision-language encoder to extract a fused CLS-token
    representation, then passes it through a 2-class linear classifier.

    Args:
        config: Configuration object with ``pretrained_model_name``
            specifying the HuggingFace model identifier.

    Attributes:
        blip: Underlying BLIP VQA model (used only for encoders).
        classifier: MLP head producing 2-class logits from the CLS token.
        return_repr: If True, ``forward()`` returns vision and text
            representations alongside logits.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.return_repr = False
        self.blip = BlipForQuestionAnswering.from_pretrained(
            config.pretrained_model_name
        )

        hidden_size = self.blip.config.text_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Forward pass for binary VQA classification.

        Args:
            input_ids: Tokenized question ``(B, seq_len)``.
            attention_mask: Attention mask ``(B, seq_len)``.
            pixel_values: Preprocessed images ``(B, 3, H, W)``.
            labels: Unused; included for API compatibility with the
                generative variant.

        Returns:
            - If ``return_repr`` is False: Dict with ``"logits"`` key
              containing ``(B, 2)`` classification logits.
            - If ``return_repr`` is True: Tuple of
              ``(text_repr, vision_repr, outputs_dict)``.
        """
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values)

        question_outputs = self.blip.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_outputs.last_hidden_state,
            encoder_attention_mask=None,
            return_dict=True,
        )

        text_repr = question_outputs.last_hidden_state[:, 0]   # (B, D)
        vision_repr = vision_outputs.last_hidden_state[:, 0]   # (B, D)

        logits = self.classifier(text_repr)
        outputs = {"logits": logits}

        if self.return_repr:
            return text_repr, vision_repr, outputs
        return outputs
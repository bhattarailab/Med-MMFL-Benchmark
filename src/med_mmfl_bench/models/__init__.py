"""Model registry and factory functions.

Provides ``get_model()`` to construct model instances by name from
configuration, and optional tokenizer factories for text models.
"""

from typing import Any

import torch.nn as nn

from med_mmfl_bench.models.symile_mimic import SymileMIMICModel

__all__ = [
    "get_model",
    "get_tokenizer",
    "SymileMIMICModel",
]

# Lazy-import registry: maps model names to (module_path, class_name).
# Models with heavy dependencies (transformers, etc.) are imported lazily
# to avoid import-time failures when those dependencies aren't installed.
_MODEL_REGISTRY = {
    "symile_mimic": ("med_mmfl_bench.models.symile_mimic", "SymileMIMICModel"),
    "rfnet": ("med_mmfl_bench.models.rfnet", "RFNet"),
    "mimic_mmclf": ("med_mmfl_bench.models.mmclf", "MultiModalClassifier"),
    "mimic_text_classifier": ("med_mmfl_bench.models.mmclf", "TextSplitClassifier"),
    "mimic_image_classifier": ("med_mmfl_bench.models.mmclf", "ImageSplitClassifier"),
    "blip_ehrxqa": ("med_mmfl_bench.models.blip", "BlipForEHRXQA"),
    "blip_yesno_vqa": ("med_mmfl_bench.models.blip", "BlipForYesNoVQA"),
}


def get_model(model_name: str, config: Any) -> nn.Module:
    """Create a model instance by name.

    Args:
        model_name: Name of the model architecture. Supported:
            ``"symile_mimic"``, ``"rfnet"``,
            ``"mimic_mmclf"``, ``"mimic_text_classifier"``,
            ``"mimic_image_classifier"``, ``"blip_ehrxqa"``,
            ``"blip_yesno_vqa"``.
        config: Model configuration object with architecture-specific
            parameters.

    Returns:
        Initialized PyTorch model.

    Raises:
        ValueError: If ``model_name`` is not recognized.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    module_path, class_name = _MODEL_REGISTRY[model_name]
    from importlib import import_module

    module = import_module(module_path)
    model_class = getattr(module, class_name)

    # All model constructors accept (config) as the first argument
    return model_class(config)


def get_tokenizer(config: Any) -> Any:
    """Create a tokenizer instance for text encoding.

    Args:
        config: Configuration with ``txt_type`` specifying the tokenizer.

    Returns:
        Initialized tokenizer.

    Raises:
        NotImplementedError: If the tokenizer type is not supported.
    """
    from transformers import BertTokenizer

    if config.txt_type == "bert-base-uncased":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    elif config.txt_type == "tiny-bert":
        return BertTokenizer.from_pretrained("huawei-noah/TinyBERT_4L_zh")
    else:
        raise NotImplementedError(
            f"Tokenizer '{config.txt_type}' is not supported."
        )

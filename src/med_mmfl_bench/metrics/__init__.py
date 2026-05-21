"""Evaluation metrics for segmentation, classification, and QA tasks.

Provides per-class metrics compatible with the ``torchmetrics`` framework
and a factory function ``get_metrics`` for constructing metric instances
by name.

Supported metrics:
    - ``dice``: Per-class Dice coefficient (segmentation)
    - ``jaccard``: Per-class Jaccard / IoU (segmentation)
    - ``accuracy``: Per-class accuracy (classification)
    - ``f1``: Per-class F1 score (classification)
    - ``qa``: Exact match, token F1, BLEU-1..4 (question answering)
"""

from typing import Any, Dict, List, Union

from med_mmfl_bench.metrics.accuracy import AccuracyPerClass
from med_mmfl_bench.metrics.dice import DicePerClass
from med_mmfl_bench.metrics.f1 import F1PerClass
from med_mmfl_bench.metrics.jaccard import JaccardPerClass
from med_mmfl_bench.metrics.qa_metrics import QAMetrics

__all__ = [
    "get_metrics",
    "DicePerClass",
    "JaccardPerClass",
    "AccuracyPerClass",
    "F1PerClass",
    "QAMetrics",
]

_METRIC_REGISTRY: Dict[str, type] = {
    "dice": DicePerClass,
    "jaccard": JaccardPerClass,
    "accuracy": AccuracyPerClass,
    "f1": F1PerClass,
    "qa": QAMetrics,
}


def get_metrics(
    eval_metrics: List[str],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a dictionary of metric instances from metric names.

    Args:
        eval_metrics: List of metric names (case-insensitive). Supported:
            ``"dice"``, ``"jaccard"``, ``"accuracy"``, ``"f1"``, ``"qa"``.
        **kwargs: Additional keyword arguments passed to each metric
            constructor (e.g., ``classes=["WT", "TC", "ET"]``).

    Returns:
        Dictionary mapping metric names to instantiated metric objects.

    Example:
        >>> metrics = get_metrics(["dice", "jaccard"], classes=["WT", "TC", "ET"])
        >>> evaluator = MetricCollection(metrics)
    """
    selected: Dict[str, Any] = {}
    for name in eval_metrics:
        key = name.lower()
        if key in _METRIC_REGISTRY:
            selected[key] = _METRIC_REGISTRY[key](**kwargs)
        else:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Available: {list(_METRIC_REGISTRY.keys())}"
            )
    return selected

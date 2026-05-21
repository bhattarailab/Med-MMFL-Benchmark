"""Text-based QA evaluation metrics: Exact Match, token F1, and BLEU-n.

Provides a ``torchmetrics``-compatible ``QAMetrics`` class that accumulates
string predictions and references, then computes Exact Match (EM),
token-level F1, and BLEU-1 through BLEU-4 at ``compute()`` time.
"""

import math
import re
from collections import Counter
from typing import Dict, List

import torch
from torchmetrics import Metric

__all__ = ["QAMetrics"]


def normalize_text(text: str) -> List[str]:
    """Lowercase, remove punctuation, and split by whitespace.

    Args:
        text: Raw input text.

    Returns:
        List of cleaned tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match between normalized prediction and reference.

    Args:
        prediction: Predicted answer string.
        reference: Ground truth answer string.

    Returns:
        1.0 if normalized tokens match exactly, 0.0 otherwise.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference.

    Args:
        prediction: Predicted answer string.
        reference: Ground truth answer string.

    Returns:
        Token-level F1 score in [0.0, 1.0].
    """
    pred_tokens = normalize_text(prediction)
    ref_tokens = normalize_text(reference)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    """Count n-grams in a token list.

    Args:
        tokens: List of word tokens.
        n: N-gram order.

    Returns:
        Counter mapping n-gram tuples to counts.
    """
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(prediction: str, reference: str, n: int) -> float:
    """Compute BLEU-n score with brevity penalty.

    Args:
        prediction: Predicted answer string.
        reference: Ground truth answer string.
        n: N-gram order (1 for BLEU-1, 4 for BLEU-4, etc.).

    Returns:
        BLEU-n score in [0.0, 1.0].
    """
    pred_tokens = normalize_text(prediction)
    ref_tokens = normalize_text(reference)

    if len(pred_tokens) == 0:
        return 0.0

    pred_ngrams = _ngram_counts(pred_tokens, n)
    ref_ngrams = _ngram_counts(ref_tokens, n)

    overlap = pred_ngrams & ref_ngrams
    num_overlap = sum(overlap.values())
    total_pred = max(sum(pred_ngrams.values()), 1)

    precision = num_overlap / total_pred

    # Brevity penalty
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))

    return bp * precision


class QAMetrics(Metric):
    """Aggregated QA evaluation metrics: EM, F1, BLEU-1..4.

    Designed for string-based predictions and references. Accumulates
    prediction/reference pairs across batches and computes all metrics
    at ``compute()`` time.

    Example:
        >>> metric = QAMetrics()
        >>> metric.update(["the heart is normal"], ["the heart is normal"])
        >>> metric.update(["lung opacity"], ["bilateral opacity"])
        >>> result = metric.compute()
        >>> result["ExactMatch"]
        0.5
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(dist_sync_on_step=False)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: List[str], targets: List[str]) -> None:
        """Accumulate a batch of prediction-reference pairs.

        Args:
            preds: List of predicted answer strings.
            targets: List of reference answer strings. Must have the
                same length as ``preds``.

        Raises:
            AssertionError: If ``preds`` and ``targets`` have different lengths.
        """
        assert len(preds) == len(targets), (
            f"Preds and targets must have same length, "
            f"got {len(preds)} and {len(targets)}"
        )
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self) -> Dict[str, float]:
        """Compute all QA metrics over accumulated pairs.

        Returns:
            Dictionary with keys: ``"ExactMatch"``, ``"F1"``,
            ``"BLEU-1"``, ``"BLEU-2"``, ``"BLEU-3"``, ``"BLEU-4"``.
        """
        em_scores: List[float] = []
        f1_scores: List[float] = []
        bleu_scores: Dict[int, List[float]] = {n: [] for n in range(1, 5)}

        for pred, ref in zip(self.preds, self.targets):
            em_scores.append(compute_exact_match(pred, ref))
            f1_scores.append(compute_f1(pred, ref))
            for n in range(1, 5):
                bleu_scores[n].append(compute_bleu(pred, ref, n))

        return {
            "ExactMatch": float(torch.tensor(em_scores).mean().item()),
            "F1": float(torch.tensor(f1_scores).mean().item()),
            "BLEU-1": float(torch.tensor(bleu_scores[1]).mean().item()),
            "BLEU-2": float(torch.tensor(bleu_scores[2]).mean().item()),
            "BLEU-3": float(torch.tensor(bleu_scores[3]).mean().item()),
            "BLEU-4": float(torch.tensor(bleu_scores[4]).mean().item()),
        }
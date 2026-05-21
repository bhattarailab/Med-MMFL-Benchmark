"""CLIP and SYMILE contrastive loss functions for multimodal learning.

Implements InfoNCE (CLIP) pairwise losses and SYMILE multilinear inner
product losses for 3-modality contrastive learning.

Reference:
    SYMILE: https://arxiv.org/abs/2309.xxxxx
"""

import itertools
from typing import List, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "clip",
    "symile",
    "infonce",
    "zeroshot_retrieval_logits",
]


def zeroshot_retrieval_logits(
    r_x: torch.Tensor,
    rep_list: List[torch.Tensor],
    logit_scale_exp: torch.Tensor,
    loss_fn: str = "clip",
) -> torch.Tensor:
    """Compute logits for zero-shot retrieval.

    Calculates retrieval logits for predicting modality ``r_x`` using
    representations in ``rep_list``, scaled by the learned temperature.

    Args:
        r_x: Encoded representations of the target modality
            ``(num_candidates, d)``.
        rep_list: List of representations for remaining modalities, each of
            shape ``(batch_sz, d)`` or ``(d,)``.
        logit_scale_exp: Exponentiated logit scale parameter (learned
            temperature).
        loss_fn: Loss function variant: ``"symile"`` or ``"clip"``.

    Returns:
        Retrieval logits of shape ``(batch_sz, num_candidates)``.
    """
    if loss_fn == "symile":
        product = torch.ones_like(rep_list[0])
        for r in rep_list:
            product = product * r
        logits = product @ r_x.t()
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

    elif loss_fn == "clip":
        for i in range(len(rep_list)):
            if rep_list[i].dim() == 1:
                rep_list[i] = rep_list[i].unsqueeze(0)

        pairwise_sum_with_r_x = torch.zeros_like(rep_list[0] @ r_x.t())
        for r in rep_list:
            pairwise_sum_with_r_x = pairwise_sum_with_r_x + r @ r_x.t()

        pairwise_sum_without_r_x = torch.zeros(
            (rep_list[0].shape[0], 1), device=rep_list[0].device
        )
        for x, y in itertools.combinations(rep_list, 2):
            pairwise_sum_without_r_x = (
                pairwise_sum_without_r_x
                + torch.diagonal(x @ y.t()).unsqueeze(dim=1)
            )

        logits = pairwise_sum_with_r_x + pairwise_sum_without_r_x
    else:
        raise ValueError(f"loss_fn must be 'symile' or 'clip', got '{loss_fn}'")

    assert logits.dim() == 2, "Logits must be a 2D tensor."
    return logit_scale_exp * logits


# ============================================================================
# CLIP (InfoNCE) Losses
# ============================================================================


def infonce(
    u: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute the CLIP (InfoNCE) loss for a pair of representations.

    Args:
        u: Representation vectors ``(batch_sz, d)``.
        v: Representation vectors ``(batch_sz, d)``.
        logit_scale: Learned temperature parameter.

    Returns:
        Scalar InfoNCE loss value.
    """
    logits_u = logit_scale * u @ v.T
    logits_v = logit_scale * v @ u.T

    assert logits_u.shape == logits_v.shape, (
        "Joint embedding spaces must have the same shape."
    )
    labels = torch.arange(logits_u.shape[0], device=u.device)
    return (
        F.cross_entropy(logits_u, labels) + F.cross_entropy(logits_v, labels)
    ) / 2.0


def clip(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    r_c: torch.Tensor,
    logit_scale: torch.Tensor,
    negative_sampling: Optional[str] = None,
) -> torch.Tensor:
    """Compute pairwise CLIP loss for three modalities.

    Calculates InfoNCE loss for all three pairs: (a,b), (b,c), (a,c).

    Args:
        r_a: Representation vectors for modality A ``(batch_sz, d)``.
        r_b: Representation vectors for modality B ``(batch_sz, d)``.
        r_c: Representation vectors for modality C ``(batch_sz, d)``.
        logit_scale: Learned temperature parameter.
        negative_sampling: Not used; included for API compatibility with
            ``symile``.

    Returns:
        Sum of the three pairwise InfoNCE losses.
    """
    loss_ab = infonce(r_a, r_b, logit_scale)
    loss_bc = infonce(r_b, r_c, logit_scale)
    loss_ac = infonce(r_a, r_c, logit_scale)
    return loss_ab + loss_bc + loss_ac


# ============================================================================
# SYMILE Losses
# ============================================================================


def compute_logits_neg_sampling_n(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Compute SYMILE logits with O(n) negative sampling.

    Draws ``n - 1`` negative samples per positive using random row shuffling
    of ``y`` and ``z``.

    Args:
        x: Anchor representation ``(batch_sz, d)``.
        y: Second modality representation ``(batch_sz, d)``.
        z: Third modality representation ``(batch_sz, d)``.

    Returns:
        Logits matrix ``(batch_sz, batch_sz)`` with positive MIPs on diagonal.
    """
    y_shuff = y[torch.randperm(y.shape[0])]
    z_shuff = z[torch.randperm(z.shape[0])]
    logits_x = x @ (y_shuff * z_shuff).t()
    mip_of_pos_triples = (x * y * z).sum(axis=1)
    eye = torch.eye(n=x.shape[0], device=x.device)
    return torch.where(eye > 0.5, mip_of_pos_triples, logits_x)


def compute_logits_neg_sampling_n_squared(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Compute SYMILE logits with O(n²) negative sampling.

    Draws ``n² - 1`` negative samples per positive by exhaustively combining
    all rows of ``y`` with cyclically shifted rows of ``z``.

    Args:
        x: Anchor representation ``(batch_sz, d)``.
        y: Second modality representation ``(batch_sz, d)``.
        z: Third modality representation ``(batch_sz, d)``.

    Returns:
        Logits matrix ``(batch_sz, batch_sz²)`` with positives on the main
        diagonal positions.
    """
    y_z = []
    z_shifted = z.clone()
    for _ in range(y.shape[0]):
        y_z.append(y * z_shifted)
        z_shifted = torch.roll(z_shifted, shifts=1, dims=0)

    y_z = torch.cat(y_z, dim=0)
    return x @ y_z.T


def symile(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    r_c: torch.Tensor,
    logit_scale: torch.Tensor,
    negative_sampling: str,
) -> torch.Tensor:
    """Compute the SYMILE loss for three modalities.

    The final loss is an average over terms where each modality serves as
    the anchor in turn.

    Args:
        r_a: Representation vectors for modality A ``(batch_sz, d)``.
        r_b: Representation vectors for modality B ``(batch_sz, d)``.
        r_c: Representation vectors for modality C ``(batch_sz, d)``.
        logit_scale: Learned temperature parameter.
        negative_sampling: Strategy for negative sampling. Must be ``"n"``
            for O(n) negatives or ``"n_squared"`` for O(n²) negatives.

    Returns:
        Scalar SYMILE loss averaged over the three anchor terms.

    Raises:
        ValueError: If ``negative_sampling`` is not ``"n"`` or ``"n_squared"``.
    """
    if negative_sampling == "n":
        compute_fn = compute_logits_neg_sampling_n
    elif negative_sampling == "n_squared":
        compute_fn = compute_logits_neg_sampling_n_squared
    else:
        raise ValueError(
            f"negative_sampling must be 'n' or 'n_squared', got '{negative_sampling}'"
        )

    logits_a = logit_scale * compute_fn(r_a, r_b, r_c)
    logits_b = logit_scale * compute_fn(r_b, r_a, r_c)
    logits_c = logit_scale * compute_fn(r_c, r_a, r_b)

    labels = torch.arange(logits_a.shape[0], device=r_a.device)
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0

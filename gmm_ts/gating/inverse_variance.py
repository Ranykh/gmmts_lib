"""Inverse-variance gating for GMM-TS.

This module implements the *uncertainty-driven* alternative to GMM-TS's learned
(transformer + MLP) gate. Instead of regressing gating weights, each expert is
weighted by the inverse of its own predictive variance ``sigma^2`` (MoGU-style):
a more confident expert (smaller variance) receives a larger weight.

Two weighting rules are provided:

* :func:`inverse_variance_weights` -- the plain ``1 / sigma^2`` softmax-normalized
  across experts. Correct when all experts share a variance scale (e.g. several
  numerical experts).

* :func:`calibrated_inverse_variance_weights` -- standardizes the log-variance
  *within each modality* before combining. This addresses the central obstacle
  of the project: a textual/LLM expert and a numerical expert emit variances on
  completely different scales, so a naive ``1 / sigma^2`` softmax collapses onto
  one modality. Per-modality normalization restores a usable gate.

The module intentionally depends only on ``torch`` so it can be imported and
unit-tested without the MM-TSFlib expert code on the path.
"""

import torch
import torch.nn.functional as F


def inverse_variance_weights(sigma2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Plain inverse-variance weights, normalized across the expert dimension.

    Args:
        sigma2: per-expert predictive variance, shape ``(B, E, H)``
            (batch, num_experts, horizon). Must be non-negative.
        eps: numerical floor to avoid division by zero.

    Returns:
        Weights of shape ``(B, E, H)`` that are non-negative and sum to 1 over
        the expert dimension ``E``.
    """
    inv = 1.0 / (sigma2 + eps)
    return inv / inv.sum(dim=1, keepdim=True)


def per_modality_log_confidence(sigma2: torch.Tensor,
                                modality_ids,
                                eps: float = 1e-8) -> torch.Tensor:
    """Confidence score from log-variance, standardized *within each modality*.

    For every modality group (e.g. all numerical experts, all textual experts)
    the log-variance is z-scored using that group's own mean and std, then
    negated so that higher = more confident. Because the standardization is
    per-modality, experts from different modalities become directly comparable
    regardless of their raw variance scale.

    Args:
        sigma2: per-expert predictive variance, shape ``(B, E, H)``.
        modality_ids: length-``E`` iterable of integer modality labels, e.g.
            ``[0, 0, 1]`` for two numerical experts and one textual expert.
        eps: numerical floor.

    Returns:
        Confidence tensor of shape ``(B, E, H)`` (unnormalized; feed to softmax).
    """
    log_v = torch.log(sigma2 + eps)
    conf = torch.zeros_like(log_v)
    ids = torch.as_tensor(list(modality_ids), device=sigma2.device)
    for m in ids.unique():
        mask = ids == m
        sub = log_v[:, mask, :]
        mean = sub.mean()
        std = sub.std() + eps
        conf[:, mask, :] = -((sub - mean) / std)
    return conf


def calibrated_inverse_variance_weights(sigma2: torch.Tensor,
                                        modality_ids,
                                        eps: float = 1e-8) -> torch.Tensor:
    """Scale-robust inverse-variance weights via per-modality normalization.

    Combines :func:`per_modality_log_confidence` with a softmax over experts.
    This is the recommended gate when mixing textual and numerical experts.

    Args:
        sigma2: per-expert predictive variance, shape ``(B, E, H)``.
        modality_ids: length-``E`` iterable of integer modality labels.
        eps: numerical floor.

    Returns:
        Weights of shape ``(B, E, H)`` summing to 1 over the expert dimension.
    """
    conf = per_modality_log_confidence(sigma2, modality_ids, eps)
    return F.softmax(conf, dim=1)

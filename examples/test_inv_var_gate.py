"""Standalone CPU test for the GMM-TS inverse-variance gate.

Validates the weighting math in ``gmm_ts/gating/inverse_variance.py`` without
requiring the MM-TSFlib experts or any dataset, and reproduces the project's
core finding: a naive ``1/sigma^2`` softmax over experts on *different variance
scales* collapses onto one modality, while per-modality normalization does not.

Run:
    python examples/test_inv_var_gate.py
"""

import os
import sys
import importlib.util

import torch

# Import the helper module directly by file path so this test does not trigger
# the gmm_ts package __init__ (which bootstraps MM-TSFlib).
HERE = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.join(HERE, "..", "gmm_ts", "gating", "inverse_variance.py")
spec = importlib.util.spec_from_file_location("inverse_variance", MOD_PATH)
iv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv)


def _degenerate_fraction(w_expert0: torch.Tensor, hi=0.95, lo=0.05) -> float:
    """Fraction of positions where the gate is ~collapsed onto one expert."""
    w = w_expert0.reshape(-1)
    return float(((w > hi) | (w < lo)).float().mean())


def test_weights_are_simplex():
    """Weights are non-negative and sum to 1 across experts."""
    torch.manual_seed(0)
    sigma2 = torch.rand(8, 3, 12) + 1e-3   # B, E, H
    w = iv.inverse_variance_weights(sigma2)
    assert torch.all(w >= 0), "weights must be non-negative"
    assert torch.allclose(w.sum(dim=1), torch.ones(8, 12), atol=1e-5), "weights must sum to 1 over experts"
    print("[ok] inverse_variance_weights lies on the simplex")


def test_lower_variance_gets_more_weight():
    """Given two experts, the lower-variance one receives the larger weight."""
    sigma2 = torch.tensor([[[0.1], [1.0]]])  # B=1, E=2, H=1
    w = iv.inverse_variance_weights(sigma2)
    assert w[0, 0, 0] > w[0, 1, 0], "lower variance expert should win"
    print(f"[ok] lower-variance expert weighted higher: {w.reshape(-1).tolist()}")


def test_scale_collapse_and_recovery():
    """Reproduce the text-vs-numeric collapse and the per-modality fix."""
    torch.manual_seed(0)
    B, H = 512, 12
    # numeric expert: O(0.2) variance; textual expert: ~100x larger scale
    var_numeric = (torch.rand(B, 1, H) * 0.2 + 0.05)
    var_textual = torch.exp(torch.randn(B, 1, H) * 0.5 + torch.log(torch.tensor(0.2))) * 100.0
    sigma2 = torch.cat([var_numeric, var_textual], dim=1)   # B, E=2, H
    modality_ids = [0, 1]                                    # numeric, textual

    w_naive = iv.inverse_variance_weights(sigma2)
    w_cal = iv.calibrated_inverse_variance_weights(sigma2, modality_ids)

    frac_naive = _degenerate_fraction(w_naive[:, 0])
    frac_cal = _degenerate_fraction(w_cal[:, 0])
    print(f"[info] naive 1/var degenerate fraction     = {frac_naive*100:5.1f}%")
    print(f"[info] per-modality degenerate fraction     = {frac_cal*100:5.1f}%")

    assert frac_naive > 0.8, "naive gate should collapse under a large scale gap"
    assert frac_cal < 0.2, "per-modality normalization should avoid collapse"
    print("[ok] naive gate collapses; per-modality normalization recovers it")


if __name__ == "__main__":
    test_weights_are_simplex()
    test_lower_variance_gets_more_weight()
    test_scale_collapse_and_recovery()
    print("\nAll inverse-variance gate tests passed.")

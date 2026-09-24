"""CPU tests for gmm_ts/gating/mogu.py against the MoGU reference implementation.

The reference formulas below are copied from yolish/moe_unc_tsf (models/MoE.py and
exp/exp_long_term_forecasting.py). No GPU, dataset or MM-TSFlib needed:

    python tests/test_mogu.py        or        python -m pytest tests/test_mogu.py
"""
import importlib.util
import os

import torch
import torch.nn as nn

# load the module by path: importing the gmm_ts package would bootstrap MM-TSFlib
_path = os.path.join(os.path.dirname(__file__), "..", "gmm_ts", "gating", "mogu.py")
_spec = importlib.util.spec_from_file_location("mogu", _path)
mogu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mogu)

B, E, H, C = 8, 3, 12, 1  # three experts: e.g. PatchTST, DLinear and an LLM


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(B, E, H, C, generator=g)
    sigma2 = torch.rand(B, E, H, C, generator=g) * 2 + 0.05
    y = torch.randn(B, H, C, generator=g)
    return mu, sigma2, y


# ---- reference (moe_unc_tsf), verbatim logic ----------------------------------------
def ref_weights(expert_unc):
    inv_var = 1.0 / (expert_unc + 1e-8)
    sum_inv_var = torch.sum(inv_var, dim=1, keepdim=True)
    return inv_var / sum_inv_var


def ref_moe_loss(outputs, expert_unc, expert_weights, batch_y):
    criterion = nn.GaussianNLLLoss(reduction='none')
    weighted_loss = None
    for i in range(outputs.shape[1]):
        expert_loss = criterion(outputs[:, i], batch_y, expert_unc[:, i])
        term = expert_loss * expert_weights[:, i]
        weighted_loss = term if weighted_loss is None else weighted_loss + term
    return weighted_loss.mean()


def ref_ale_epi(outputs, agg_outputs, expert_unc, expert_weights):
    aleatoric = torch.sum(expert_unc * expert_weights, dim=1)
    epistemic = None
    for i in range(outputs.shape[1]):
        d = expert_weights[:, i] * (agg_outputs - outputs[:, i]) ** 2
        epistemic = d if epistemic is None else epistemic + d
    return aleatoric, epistemic


# ---- tests --------------------------------------------------------------------------
def test_weights_match_reference_and_sum_to_one():
    _, sigma2, _ = _inputs()
    w = mogu.inverse_variance_weights(sigma2)
    assert torch.allclose(w, ref_weights(sigma2))
    assert torch.allclose(w.sum(dim=1), torch.ones(B, H, C))


def test_more_confident_expert_gets_more_weight():
    sigma2 = torch.ones(1, E, 1, 1)
    assert torch.allclose(mogu.inverse_variance_weights(sigma2), torch.full_like(sigma2, 1 / E))
    sigma2[0, 1] = 0.1  # expert 1 is 10x more confident
    w = mogu.inverse_variance_weights(sigma2)[0, :, 0, 0]
    assert w.argmax().item() == 1 and torch.allclose(w, torch.tensor([1, 10, 1]) / 12.0)


def test_loss_matches_reference():
    mu, sigma2, y = _inputs(1)
    w = mogu.inverse_variance_weights(sigma2)
    assert torch.allclose(mogu.mogu_loss(mu, sigma2, w, y), ref_moe_loss(mu, sigma2, w, y))


def test_gate_is_not_detached():
    """MoGU trains the variances through the gate as well as through each expert's NLL."""
    mu, sigma2, y = _inputs(2)
    sigma2.requires_grad_(True)
    mogu.mogu_loss(mu, sigma2, mogu.inverse_variance_weights(sigma2), y).backward()
    grad_full = sigma2.grad.clone()
    sigma2.grad = None
    w_detached = mogu.inverse_variance_weights(sigma2).detach()
    mogu.mogu_loss(mu, sigma2, w_detached, y).backward()
    assert not torch.allclose(grad_full, sigma2.grad)


def test_aleatoric_epistemic_match_reference():
    mu, sigma2, _ = _inputs(3)
    w = mogu.inverse_variance_weights(sigma2)
    ale, epi = mogu.aleatoric_epistemic(mu, sigma2, w)
    ref_ale, ref_epi = ref_ale_epi(mu, (w * mu).sum(dim=1), sigma2, w)
    assert torch.allclose(ale, ref_ale) and torch.allclose(epi, ref_epi)


def test_uncertainty_head_shape_and_sign():
    torch.manual_seed(0)
    for arc in ("mlp", "linear"):
        head = mogu.UncertaintyHead(in_dim=20, pred_len=H, arc_type=arc)
        s2 = head(torch.randn(B, 20))
        assert s2.shape == (B, H, 1) and bool((s2 >= 0).all())


def test_zero_variance_is_guarded_like_the_reference():
    """softplus can underflow to exactly 0 in float32; MoGU guards it with +1e-8 in the
    weights and the NLL's variance clamp, so the gate and the loss stay finite."""
    mu, sigma2, y = _inputs(4)
    sigma2[:, 0] = 0.0
    w = mogu.inverse_variance_weights(sigma2)
    assert torch.isfinite(w).all() and torch.allclose(w.sum(dim=1), torch.ones(B, H, C))
    assert torch.isfinite(mogu.mogu_loss(mu, sigma2, w, y))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print("{} passed".format(len(tests)))

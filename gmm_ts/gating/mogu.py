"""MoGU gating for GMM-TS: weight each expert by the inverse of its predicted variance.

A faithful port of the MoGU reference implementation (yolish/moe_unc_tsf) into GMM-TS:

  moe_unc_tsf                                         here
  --------------------------------------------------  ---------------------------
  models/iTransformer.py        UncHead               UncertaintyHead
  models/MoE.py                 unc_gating branch     inverse_variance_weights
  exp/exp_long_term_forecasting moe_loss              mogu_loss
  exp/exp_long_term_forecasting calc_aleatoric_...    aleatoric_epistemic

The gate has no parameters: w_e = (1/sigma_e^2) / sum_j (1/sigma_j^2), per horizon step,
for any number of experts E >= 2. Training minimises the gate-weighted Gaussian NLL of the
experts, sum_e w_e * NLL(mu_e, sigma_e^2; y), exactly as MoGU does -- the weights are not
detached, so the variance heads are trained both by each expert's NLL and through the gate.

Tensor layout throughout: (B, E, H, C) = batch, experts, horizon, channels (C = 1 in GMM-TS).

Torch-only on purpose: importable and unit-testable without MM-TSFlib.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    """MoGU's UncHead: features -> sigma^2 over the forecast horizon.

    Same architecture choices as the reference: 'mlp' = Linear(d, d) -> ReLU -> Linear(d, H),
    'linear' = Linear(d, H); softplus(threshold=20) keeps sigma^2 positive.
    """

    def __init__(self, in_dim: int, pred_len: int, arc_type: str = "mlp"):
        super().__init__()
        if arc_type == "linear":
            self.net = nn.Linear(in_dim, pred_len)
        elif arc_type == "mlp":
            self.net = nn.Sequential(nn.Linear(in_dim, in_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(in_dim, pred_len))
        else:
            raise ValueError("unc_head_type must be 'mlp' or 'linear', got {}".format(arc_type))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, in_dim) -> sigma^2: (B, pred_len, 1)."""
        return F.softplus(self.net(h), threshold=20).unsqueeze(-1)


def inverse_variance_weights(sigma2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B, E, H, C) variances -> (B, E, H, C) weights that sum to 1 over E (MoGU MoE.py)."""
    inv_var = 1.0 / (sigma2 + eps)
    return inv_var / inv_var.sum(dim=1, keepdim=True)


def mogu_loss(mu: torch.Tensor, sigma2: torch.Tensor, weights: torch.Tensor,
              y: torch.Tensor) -> torch.Tensor:
    """MoGU training loss: mean over (B, H, C) of sum_e w_e * GaussianNLL(mu_e, y, sigma2_e).

    mu, sigma2, weights: (B, E, H, C); y: (B, H, C). Same values as the reference's
    per-expert nn.GaussianNLLLoss(reduction='none') loop (full=False, eps=1e-6).
    """
    nll = F.gaussian_nll_loss(mu, y.unsqueeze(1).expand_as(mu), sigma2, reduction="none")
    return (nll * weights).sum(dim=1).mean()



def mogu_mse_loss(mu: torch.Tensor, sigma2: torch.Tensor, weights: torch.Tensor,
                  y: torch.Tensor, n_numeric: int) -> torch.Tensor:
    """Inverse-variance gate on experts trained with GMM-TS's objective (the `mogu_mse` arm).

    Forecasts get exactly the terms vanilla GMM-TS trains with -- MSE of the gated forecast
    plus each numeric expert's own MSE -- with the gate weights detached, so no forecast is
    trained to win or lose weight. Variance heads get only the Gaussian NLL of their expert's
    detached forecast, unweighted, so each sigma^2 is fitted to its own expert's error.
    Gradients therefore separate cleanly: forecasts <- MSE terms, heads <- NLL term.

    mu, sigma2, weights: (B, E, H, C) with the numeric experts first; y: (B, H, C).
    """
    y_hat = (weights.detach() * mu).sum(dim=1)
    loss = F.mse_loss(y_hat, y)
    for e in range(n_numeric):
        loss = loss + F.mse_loss(mu[:, e], y)
    nll = F.gaussian_nll_loss(mu.detach(), y.unsqueeze(1).expand_as(mu), sigma2, reduction="none")
    return loss + nll.mean(dim=(0, 2, 3)).sum()

def aleatoric_epistemic(mu: torch.Tensor, sigma2: torch.Tensor, weights: torch.Tensor):
    """MoGU's uncertainty split of the mixture. Inputs (B, E, H, C); outputs (B, H, C).

    aleatoric = sum_e w_e sigma_e^2          (expected noise the experts report)
    epistemic = sum_e w_e (y_hat - mu_e)^2   (disagreement between experts)
    """
    y_hat = (weights * mu).sum(dim=1, keepdim=True)
    aleatoric = (weights * sigma2).sum(dim=1)
    epistemic = (weights * (y_hat - mu) ** 2).sum(dim=1)
    return aleatoric, epistemic

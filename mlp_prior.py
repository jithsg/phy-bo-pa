"""Cheap-fidelity MLP prior functions for constraint GP means.

Loads results/sparam_mlp.pt (trained on 1,400 SP-only simulations; Stage-II
cross-fidelity R^2: S11 0.73, S22 0.80, logK 0.997, B1 0.98) and exposes
differentiable functions of the raw 7-D design vector for use inside
PhysicsMean. Constraint-local indices: 0=K, 1=B1, 2=S11_dB, 3=S22_dB.

Returns None when the checkpoint is absent, so the BO loop falls back to the
original priors transparently.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from config import BOUNDS_LOWER, BOUNDS_UPPER

CKPT = Path(__file__).parent / "results" / "sparam_mlp.pt"
_STATE: dict = {}


def _net():
    if "net" not in _STATE:
        from train_sparam_prior import make_mlp
        ckpt = torch.load(CKPT, weights_only=False)
        net = make_mlp()
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _STATE["net"] = net
        _STATE["mu"] = torch.tensor(ckpt["y_mu"], dtype=torch.double)
        _STATE["sd"] = torch.tensor(ckpt["y_sd"], dtype=torch.double)
        _STATE["lo"] = torch.tensor(BOUNDS_LOWER, dtype=torch.double)
        _STATE["hi"] = torch.tensor(BOUNDS_UPPER, dtype=torch.double)
    return _STATE


def _predict(x_raw: Tensor, out_idx: int) -> Tensor:
    s = _net()
    xn = (x_raw - s["lo"]) / (s["hi"] - s["lo"])
    out = s["net"](xn)[..., out_idx]
    return out * s["sd"][out_idx] + s["mu"][out_idx]


def mlp_constraint_fns() -> dict[int, object] | None:
    """Constraint-local index -> differentiable raw-value predictor."""
    if not CKPT.exists():
        return None
    return {
        0: lambda x: torch.pow(10.0, torch.clamp(_predict(x, 2), max=6.0)),  # K
        1: lambda x: _predict(x, 3),   # B1
        2: lambda x: _predict(x, 0),   # S11_dB
        3: lambda x: _predict(x, 1),   # S22_dB
    }

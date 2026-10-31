"""πBO-style prior weighting for the PA MOBO loop (Hvarfner et al., ICLR 2022).

The decaying prior-weighted acquisition of πBO (paper Eq. 6, Algorithm 1 line 6) is

    α_{π,n}(x) = α(x, D_{n-1}) · π(x)^{β/n},      n = 1, 2, ..., N  (BO iteration),

so for a *log* acquisition such as qLogNEHVI the one-line change is

    log α_{π,n}(x) = log α(x) + (β/n) · log π(x).

Conventions taken from the paper:
  * n is the 1-indexed BO iteration after the initial design (Algorithm 1).
  * β = N/10 with N the number of BO iterations (Section 5, "Experimental setup").
  * π is a Gaussian centred on the pre-existing default design with a standard
    deviation of 25 % of each variable's domain (Section 5), truncated to the
    search space, plus a small constant ε = 1e-12 so that π is strictly positive
    everywhere (Appendix F.1).

The prior is built from `config.PRESIM_X` (the manual design that existed before
any BO run) and the variable bounds only.  Nothing in this module reads a result
file.  This is a multi-objective extension of the πBO idea on top of constrained
qLogNEHVI; the original πBO code (single-objective EI in Spearmint/HyperMapper)
is not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction

from config import BOUNDS_LOWER, BOUNDS_UPPER, N_DIMS, PRESIM_X

DTYPE = torch.double


def _log_ndtr_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log(Φ(b) − Φ(a)) for a < b, elementwise."""
    normal = torch.distributions.Normal(0.0, 1.0)
    return torch.log(normal.cdf(b) - normal.cdf(a))


@dataclass(frozen=True)
class PriorSpec:
    """Independent truncated-Gaussian prior over the normalised design space [0,1]^d."""

    center: tuple[float, ...]        # normalised coordinates of the prior mode
    sigma: float = 0.25              # std as a fraction of each variable's domain
    eps: float = 1e-12               # additive floor (πBO Appendix F.1)
    beta: float = 16.0               # prior confidence, β = N/10 for N = 160

    @classmethod
    def manual_design(cls, n_iterations: int = 160, sigma: float = 0.25,
                      eps: float = 1e-12) -> "PriorSpec":
        """Prior centred on the manual (pre-BO) design of config.PRESIM_X."""
        c = (np.asarray(PRESIM_X) - BOUNDS_LOWER) / (BOUNDS_UPPER - BOUNDS_LOWER)
        if not np.all((c >= 0.0) & (c <= 1.0)):
            raise ValueError(f"manual design outside the bounds in normalised space: {c}")
        return cls(center=tuple(float(v) for v in c), sigma=sigma, eps=eps,
                   beta=n_iterations / 10.0)

    @classmethod
    def uniform(cls, n_iterations: int = 160) -> "PriorSpec":
        """Uniform prior (log π ≡ 0): must reproduce plain qLogNEHVI exactly."""
        return cls(center=tuple([0.5] * N_DIMS), sigma=math.inf, eps=0.0,
                   beta=n_iterations / 10.0)

    # ---- density -------------------------------------------------------
    def log_prior(self, X: torch.Tensor) -> torch.Tensor:
        """log π(x) for X of shape (..., d) in normalised coordinates.

        π(x) = Π_d TN(x_d; c_d, σ², [0, 1]) + ε, a proper density on [0,1]^d
        (up to the ε mass) that is strictly positive everywhere.
        """
        if math.isinf(self.sigma):
            return torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        c = torch.tensor(self.center, dtype=X.dtype, device=X.device)
        s = self.sigma
        z = (X - c) / s
        log_phi = -0.5 * z ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)
        lo = (torch.zeros_like(c) - c) / s
        hi = (torch.ones_like(c) - c) / s
        log_norm = _log_ndtr_diff(lo, hi)                     # per-dim truncation
        log_gauss = (log_phi - log_norm).sum(dim=-1)
        if self.eps > 0.0:
            log_eps = torch.full_like(log_gauss, math.log(self.eps))
            return torch.logaddexp(log_gauss, log_eps)
        return log_gauss

    def weight(self, n: int) -> float:
        """Exponent γ_n = β/n of the prior at BO iteration n (1-indexed)."""
        if n < 1:
            raise ValueError("πBO iteration index n starts at 1")
        return self.beta / n

    def as_dict(self) -> dict:
        return {"center_norm": list(self.center), "center_raw": list(map(float, PRESIM_X)),
                "sigma_frac": self.sigma, "eps": self.eps, "beta": self.beta,
                "decay": "gamma_n = beta / n, n = 1..N (BO iteration after the DoE)",
                "source": "config.PRESIM_X and config.BOUNDS_* only"}


class PriorWeightedAcqf(AcquisitionFunction):
    """log α(X) + γ_n · Σ_q log π(x_q) around any log-acquisition function."""

    def __init__(self, base: AcquisitionFunction, prior: PriorSpec, n: int):
        super().__init__(model=base.model)
        self.base = base
        self.prior = prior
        self.gamma = prior.weight(n)

    def forward(self, X: torch.Tensor) -> torch.Tensor:          # X: (b, q, d)
        base_val = self.base(X)
        lp = self.prior.log_prior(X).sum(dim=-1)                 # sum over q
        return base_val + self.gamma * lp

    # optimize_acqf bookkeeping goes to the wrapped acquisition
    def set_X_pending(self, X_pending=None):
        self.base.set_X_pending(X_pending)

    @property
    def X_pending(self):
        return getattr(self.base, "X_pending", None)

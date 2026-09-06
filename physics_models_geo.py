"""Physics-informed prior mean functions for PA direct geometry optimization.

Adapted from physics_models.py (library-based) to use direct geometry inputs:
  x = [L1L2_rad, L1L2_nr, L3L4_rad, L3L4_nr, C_wr, R_l, M_wr]

Key change: inductor L and Q are computed from (rad, nr) via modified Wheeler
approximation (K1=1.754, K2=-0.410, calibrated to TSMC 40nm spiral_std_mu_z)
instead of table lookup.

Physics models:
  - Gain:  gm * Z_load (calibrated)
  - S11:   Input matching from L1L2 + Cgs
  - S22:   Output matching from L3L4 + Cds
  - Pdc:   Vdd * Id (Id ∝ M_wr)

The GP learns: y_spectre = physics_mean(x) + GP_residual(x)
"""

import math
from dataclasses import dataclass

import gpytorch
import torch
from botorch.utils.transforms import unnormalize
from torch import Tensor

from config import IND_W, IND_SPACING

# ---------------------------------------------------------------
# Physics constants (from bo_pa/config.py PAPhysicsConstants)
# ---------------------------------------------------------------

@dataclass(frozen=True)
class PAPhysicsConstants:
    """Physical constants for PA analytical models.

    Calibrated against TSMC 40nm push-pull PA pre-sim baseline.
    """
    VDD: float = 1.1
    VBIAS: float = 0.5
    f0: float = 2.4e9
    Z0: float = 50.0

    # MOSFET (nmos_rf lr=40n, multi=100)
    mos_lr: float = 40e-9
    mos_multi: int = 100
    mu_n_Cox: float = 300e-6       # A/V^2 for 40nm
    Vth: float = 0.4

    # Calibration factors (from pre-sim baseline)
    k_gain: float = 0.36
    k_pae: float = 1.63
    k_psat: float = 2.85
    k_Cgs: float = 8.0
    k_Cds: float = 25.0
    k_Rin: float = 5.0
    k_Rout: float = 1.0


PA_PHYSICS = PAPhysicsConstants()

# ---------------------------------------------------------------
# Direct geometry input indices
# ---------------------------------------------------------------
_I_L1L2_RAD = 0   # um
_I_L1L2_NR = 1
_I_L3L4_RAD = 2   # um
_I_L3L4_NR = 3
_I_C_WR = 4       # meters
_I_R_L = 5        # meters
_I_M_WR = 6       # meters

# ---------------------------------------------------------------
# Modified Wheeler inductance model
# ---------------------------------------------------------------
# K1, K2 calibrated against a small set of foundry EM characterizations
# (reference data withheld under the PDK NDA; calibrated values are public).
# Mean abs error: 5.5% vs Spectre (original Wheeler: 15%).

_MU0 = 4.0 * math.pi * 1e-7  # H/m
_WHEELER_K1 = 1.754
_WHEELER_K2 = -0.410
_W_UM = IND_W         # PDK trace width (withheld; see pdk_private.example.py)
_SP_UM = IND_SPACING  # PDK turn spacing (withheld; see pdk_private.example.py)


def _wheeler_L_nH(rad_um: Tensor, nr: Tensor) -> Tensor:
    """Modified Wheeler inductance (nH) from spiral geometry.

    L = K1 * mu0 * n^2 * d_avg / (1 + K2 * rho)

    Args:
        rad_um: outer radius in um
        nr: number of turns
    """
    d_out = 2.0 * rad_um * 1e-6   # m
    d_in = d_out - 2.0 * nr * (_W_UM + _SP_UM) * 1e-6  # m
    d_in = torch.clamp(d_in, min=1e-6)  # safety
    d_avg = (d_out + d_in) / 2.0
    rho = (d_out - d_in) / (d_out + d_in + 1e-12)
    L = _WHEELER_K1 * _MU0 * nr ** 2 * d_avg / (1.0 + _WHEELER_K2 * rho)
    return L * 1e9  # nH


def _approx_Q(rad_um: Tensor, nr: Tensor) -> Tensor:
    """Approximate inductor Q from geometry.

    Q degrades with more turns (higher loss) and smaller radius (more crowding).
    Simple empirical model calibrated to the characterized Q range:
      Q ≈ Q_base - k_nr * nr - k_fill * rho
    """
    d_out = 2.0 * rad_um
    d_in = d_out - 2.0 * nr * (_W_UM + _SP_UM)
    d_in = torch.clamp(d_in, min=1.0)
    rho = (d_out - d_in) / (d_out + d_in + 1e-6)

    # Empirical: Q peaks at moderate fill, degrades with high nr
    Q = 14.0 - 0.5 * nr - 3.0 * rho
    return torch.clamp(Q, min=5.0, max=15.0)


# ---------------------------------------------------------------
# MOSFET helpers (same as physics_models.py)
# ---------------------------------------------------------------

def _gm(m_wr: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    W_total = m_wr * pc.mos_multi
    Vov = max(pc.VBIAS - pc.Vth, 0.05)
    return pc.mu_n_Cox * (W_total / pc.mos_lr) * Vov


def _Cgs(m_wr: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    W_total = m_wr * pc.mos_multi
    Cox_area = 10e-3   # F/m^2
    Cov_per_w = 0.3e-6  # F/m
    C_intrinsic = (2.0 / 3.0) * Cox_area * W_total * pc.mos_lr + Cov_per_w * W_total
    return pc.k_Cgs * C_intrinsic


def _Cds(m_wr: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    W_total = m_wr * pc.mos_multi
    Cov_per_w = 0.2e-6  # F/m
    return pc.k_Cds * Cov_per_w * W_total


# ---------------------------------------------------------------
# Physics functions — direct geometry versions
# ---------------------------------------------------------------

def physics_gain(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """Small-signal gain (dB). Gain = 20*log10(k * gm * Z_load)."""
    m_wr = x_raw[..., _I_M_WR]
    gm = _gm(m_wr, pc)

    # Drain inductor from geometry
    L3_nH = _wheeler_L_nH(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    Q3 = _approx_Q(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    omega = 2.0 * math.pi * pc.f0
    X_L = omega * L3_nH * 1e-9
    R_p = Q3 * X_L  # parallel resistance

    Z_load = R_p * pc.Z0 / (R_p + pc.Z0)
    A_v = pc.k_gain * gm * Z_load

    return 20.0 * torch.log10(torch.clamp(A_v, min=0.1))


def physics_s11(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """Input return loss S11 (dB) from impedance matching.

    Zin = R_loss + j(ωL1 - 1/(ωCgs))
    S11 = 20*log10(|Zin - Z0| / |Zin + Z0|)
    """
    m_wr = x_raw[..., _I_M_WR]
    omega = 2.0 * math.pi * pc.f0

    # Input inductor from geometry
    L1_nH = _wheeler_L_nH(x_raw[..., _I_L1L2_RAD], x_raw[..., _I_L1L2_NR])
    Q1 = _approx_Q(x_raw[..., _I_L1L2_RAD], x_raw[..., _I_L1L2_NR])
    L1 = L1_nH * 1e-9
    X_L1 = omega * L1
    R_L1 = X_L1 / Q1

    # Gate capacitance
    Cgs = _Cgs(m_wr, pc)
    X_Cgs = 1.0 / (omega * torch.clamp(Cgs, min=1e-15))

    # Input impedance (series RLC)
    R_in = pc.k_Rin * (R_L1 + 0.5)
    X_in = X_L1 - X_Cgs

    # S11
    num_mag2 = (R_in - pc.Z0) ** 2 + X_in ** 2
    den_mag2 = (R_in + pc.Z0) ** 2 + X_in ** 2
    s11_mag2 = num_mag2 / torch.clamp(den_mag2, min=1e-10)

    return 10.0 * torch.log10(torch.clamp(s11_mag2, min=1e-10))


def physics_s22(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """Output return loss S22 (dB) from impedance matching.

    Zout = R_loss + j(ωL3 - 1/(ωCds))
    S22 = 20*log10(|Zout - Z0| / |Zout + Z0|)
    """
    m_wr = x_raw[..., _I_M_WR]
    omega = 2.0 * math.pi * pc.f0

    # Drain inductor from geometry
    L3_nH = _wheeler_L_nH(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    Q3 = _approx_Q(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    L3 = L3_nH * 1e-9
    X_L3 = omega * L3
    R_L3 = X_L3 / Q3

    # Drain capacitance
    Cds = _Cds(m_wr, pc)
    X_Cds = 1.0 / (omega * torch.clamp(Cds, min=1e-15))

    # Output impedance
    R_out = pc.k_Rout * (R_L3 + 1.0)
    X_out = X_L3 - X_Cds

    # S22
    num_mag2 = (R_out - pc.Z0) ** 2 + X_out ** 2
    den_mag2 = (R_out + pc.Z0) ** 2 + X_out ** 2
    s22_mag2 = num_mag2 / torch.clamp(den_mag2, min=1e-10)

    return 10.0 * torch.log10(torch.clamp(s22_mag2, min=1e-10))


def physics_pdc(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """DC power consumption (mW). Pdc = Vdd * Id, Id ∝ gm * Vov."""
    m_wr = x_raw[..., _I_M_WR]
    gm = _gm(m_wr, pc)
    Vov = max(pc.VBIAS - pc.Vth, 0.05)
    I_dc = pc.k_pae * gm * Vov * 0.5
    P_dc = I_dc * pc.VDD
    return P_dc * 1e3  # mW


def physics_psat(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """Saturated output power (dBm).

    Class AB: Psat ≈ k * Vdd^2 / (2 * R_load)
    R_load = R_p || Z0, where R_p = Q * ωL from drain inductor.
    """
    L3_nH = _wheeler_L_nH(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    Q3 = _approx_Q(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    omega = 2.0 * math.pi * pc.f0
    X_L = omega * L3_nH * 1e-9
    R_p = Q3 * X_L
    R_load = R_p * pc.Z0 / (R_p + pc.Z0)

    P_sat_W = pc.k_psat * pc.VDD ** 2 / (2.0 * torch.clamp(R_load, min=1.0))
    P_sat_dBm = 10.0 * torch.log10(torch.clamp(P_sat_W * 1e3, min=1e-6))
    return P_sat_dBm


def physics_oip3(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """Output third-order intercept point (dBm).

    OIP3 ≈ Psat + backoff, where backoff ~ 10 dB for Class AB.
    Also scales with Vov (higher overdrive → better linearity):
      OIP3 ≈ 10*log10(k * gm * Vov^2 * R_load) + 30
    """
    m_wr = x_raw[..., _I_M_WR]
    gm = _gm(m_wr, pc)
    Vov = max(pc.VBIAS - pc.Vth, 0.05)

    L3_nH = _wheeler_L_nH(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    Q3 = _approx_Q(x_raw[..., _I_L3L4_RAD], x_raw[..., _I_L3L4_NR])
    omega = 2.0 * math.pi * pc.f0
    X_L = omega * L3_nH * 1e-9
    R_p = Q3 * X_L
    R_load = R_p * pc.Z0 / (R_p + pc.Z0)

    # OIP3 in watts, then convert to dBm
    # k_oip3 ~ 4/3 for weakly nonlinear Class AB
    k_oip3 = 1.33
    P_oip3_W = k_oip3 * gm * Vov ** 2 * R_load
    P_oip3_dBm = 10.0 * torch.log10(torch.clamp(P_oip3_W * 1e3, min=1e-6))
    return P_oip3_dBm


# ---------------------------------------------------------------
# Empirical S11 matching model
# ---------------------------------------------------------------

def physics_s11(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """S11 (dB) — empirical matching score model.

    S11 ≈ -5.8 * log10(ω² * L1 * Cgs_simple * c_wr / r_l) - 10.6

    Based on linear regression of match_in score vs Spectre S11
    across 100 designs (R²=0.21, MAE=3.1 dB, correlation=-0.46).
    Captures the monotonic trend: higher match_in → deeper S11.
    Uses simplified Cgs = (2/3)*Cox*W*L to match calibration data.
    """
    L1_nH = _wheeler_L_nH(x_raw[..., _I_L1L2_RAD], x_raw[..., _I_L1L2_NR])
    L1 = L1_nH * 1e-9  # H

    # Simplified Cgs matching the calibration formula
    # Cgs = (2/3) * Cox * W_total * Lr, Cox=8.95e-3 F/m^2
    m_wr = x_raw[..., _I_M_WR]  # m
    W_total = m_wr * pc.mos_multi  # m
    Cgs = (2.0 / 3.0) * 8.95e-3 * W_total * pc.mos_lr  # F

    c_wr = x_raw[..., _I_C_WR]  # m
    r_l = x_raw[..., _I_R_L]    # m

    omega = 2.0 * math.pi * pc.f0
    match_in = omega ** 2 * L1 * Cgs * c_wr / torch.clamp(r_l, min=1e-9)

    log_match = torch.log10(torch.clamp(match_in, min=1e-10))
    s11_pred = -5.801 * log_match - 10.593
    return s11_pred


# ---------------------------------------------------------------
# Empirical S22 output matching model
# ---------------------------------------------------------------

def physics_s22(x_raw: Tensor, pc: PAPhysicsConstants = PA_PHYSICS) -> Tensor:
    """S22 (dB) — empirical output matching model.

    S22 ≈ -1.404 * m_wr_um - 0.609 * c_wr_um - 2.169

    Based on linear regression across 174 designs (R²=0.10, MAE=3.2 dB).
    Captures: larger device (lower Rout) + larger coupling cap → deeper S22.
    """
    m_wr_um = x_raw[..., _I_M_WR] * 1e6   # convert m → um
    c_wr_um = x_raw[..., _I_C_WR] * 1e6   # convert m → um
    s22_pred = -1.4043 * m_wr_um - 0.6090 * c_wr_um - 2.169
    return s22_pred


# ---------------------------------------------------------------
# Which outputs get physics priors
# ---------------------------------------------------------------

# Objectives: [PAE, Gain, Psat]
OBJECTIVE_PHYSICS = {
    1: physics_gain,    # Gain (dB)
    2: physics_psat,    # Psat (dBm)
}

# Constraints (index in CONSTRAINT_TRANSFORMS):
# 0=K, 1=B1, 2=S11, 3=S22, 4=OIP3, 5=Pdc
CONSTRAINT_PHYSICS = {
    2: physics_s11,     # S11 (dB) — empirical matching score model
    3: physics_s22,     # S22 (dB) — empirical output matching model
    4: physics_oip3,    # OIP3 (dBm)
    5: physics_pdc,     # Pdc (mW) — needs sign-flip transform
}


# ---------------------------------------------------------------
# GPyTorch Mean module (same interface as physics_models.py)
# ---------------------------------------------------------------

class PhysicsMean(gpytorch.means.Mean):
    """Physics-informed GP prior mean function with learnable strength.

    The standardized physics prediction passes through a learnable affine map
    m(x) = a * phi(x) + b (a init 1, b init 0), fitted jointly with the kernel
    hyperparameters by marginal-likelihood maximization. A misspecified prior
    is thereby down-weighted toward a constant mean as evidence accumulates,
    instead of steering the GP at full fixed strength (the cold-start failure
    mode identified in the attribution study).
    """

    def __init__(self, physics_fn, bounds, transform_fn=None):
        super().__init__()
        self.register_buffer("bounds", bounds)
        self.transform_fn = transform_fn
        self._physics_fn = physics_fn
        self.register_buffer("y_mean", torch.tensor(0.0, dtype=torch.double))
        self.register_buffer("y_std", torch.tensor(1.0, dtype=torch.double))
        self._standardized = False
        self.register_parameter(
            "prior_scale",
            torch.nn.Parameter(torch.tensor(1.0, dtype=torch.double)))
        self.register_parameter(
            "prior_offset",
            torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double)))

    def set_standardization(self, y_mean: float, y_std: float) -> None:
        self.y_mean.fill_(y_mean)
        self.y_std.fill_(max(y_std, 1e-8))
        self._standardized = True

    def forward(self, x: Tensor) -> Tensor:
        # Only use first 7 dims (raw geometry) — ignore appended physics features
        n_raw = self.bounds.shape[-1]
        x_raw = unnormalize(x[..., :n_raw], self.bounds)
        y_raw = self._physics_fn(x_raw)

        if self.transform_fn is not None:
            y_raw = self.transform_fn(y_raw)

        if self._standardized:
            y_raw = (y_raw - self.y_mean) / self.y_std
        return self.prior_scale * y_raw + self.prior_offset


def create_objective_physics_means(bounds: Tensor) -> dict[int, PhysicsMean]:
    """Create PhysicsMean for objectives that have physics models.

    Returns dict: output_index -> PhysicsMean.
    """
    return {
        idx: PhysicsMean(fn, bounds)
        for idx, fn in OBJECTIVE_PHYSICS.items()
    }


def create_constraint_physics_means(
    bounds: Tensor,
    constraint_transforms: list,
) -> dict[int, PhysicsMean]:
    """Create PhysicsMean for constraints that have physics models.

    Applies the constraint transform (scale, offset) so the mean
    predicts in the same space as the GP target.
    """
    means = {}
    for idx, physics_fn in CONSTRAINT_PHYSICS.items():
        _, scale, offset = constraint_transforms[idx]

        def make_transform(s=scale, o=offset):
            def transform(y_raw):
                return y_raw * s + o
            return transform

        means[idx] = PhysicsMean(
            physics_fn, bounds, transform_fn=make_transform(),
        )
    return means

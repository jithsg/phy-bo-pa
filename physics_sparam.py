"""Physics-structured S11/S22 priors from circuit analysis of the PA netlist.

Replaces the weak linear-regression S-parameter priors (R^2 0.08-0.30) with
closed-form small-signal models of the actual matching topology (per
differential half, at f0 = 2.4 GHz):

  input  : series coupling moscap C_c(c_wr) -> shunt gate spiral
           L_g(r1, n1) to AC ground -> series bias resistor R_b(r_l) ->
           device input capacitance C_g(W_tot);
  output : drain spiral L_d(r3, n3) to VDD in parallel with device output
           capacitance C_d(W_tot) and r_o(W_tot), seen through the output
           balun (effective impedance ratio k_bal).

Element values follow the modified-Wheeler inductance and simple area/width
scalings whose coefficients (theta, 7 values) are PDK device constants,
calibrated once by least squares on an evaluation corpus. The model is
differentiable in the 7 design variables, so it can serve directly as a
PhysicsMean function.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor

F0_HZ = 2.4e9
OMEGA = 2.0 * math.pi * F0_HZ
Z0 = 50.0
W_TOT_FACTOR = 100.0  # MOS_MULTI: total width = m_wr[um] * 100

THETA_NAMES = ("rho_L", "k_cap", "k_res", "k_cgs", "k_cds", "k_ro", "k_bal")
# Initial guesses in engineering units:
#   rho_L  [ohm/um]   spiral wire resistance per unit length
#   k_cap  [fF/um^2]  moscap area capacitance
#   k_res  [ohm/um]   bias resistor per unit length
#   k_cgs  [fF/um]    device input capacitance per um width
#   k_cds  [fF/um]    device output capacitance per um width
#   k_ro   [ohm*um]   output resistance * width
#   k_bal  [-]        effective output impedance transform ratio
THETA_INIT = (0.05, 4.0, 50.0, 1.0, 0.6, 3.0e5, 0.5)
THETA_PATH = Path(__file__).parent / "results" / "sparam_prior_theta.json"


def _wheeler_L_H(rad_um: Tensor, nr: Tensor) -> Tensor:
    """Modified Wheeler inductance [H] (mirrors physics_models_geo)."""
    w_um, s_um = 6.0, 2.0
    d_out = 2.0 * rad_um
    d_in = d_out - 2.0 * nr * (w_um + s_um)
    d_in = torch.clamp(d_in, min=1.0)
    d_avg = 0.5 * (d_out + d_in)
    rho = (d_out - d_in) / torch.clamp(d_out + d_in, min=1e-9)
    k1, k2 = 2.34, 2.75
    mu0 = 4.0e-7 * math.pi
    L = k1 * mu0 * nr**2 * (d_avg * 1e-6) / (1.0 + k2 * rho)
    return L


def _series_parallel(*z: Tensor) -> Tensor:
    y = sum(1.0 / zi for zi in z)
    return 1.0 / y


def s11_db_model(x_raw: Tensor, theta: dict[str, Tensor]) -> Tensor:
    """S11 [dB] at f0 from the input-match chain. x_raw: (..., 7) raw units."""
    r1, n1 = x_raw[..., 0], x_raw[..., 1]
    c_wr_um = x_raw[..., 4] * 1e6
    r_l_um = x_raw[..., 5] * 1e6
    w_tot_um = x_raw[..., 6] * 1e6 * W_TOT_FACTOR

    L_g = _wheeler_L_H(r1, n1)
    R_Lg = theta["rho_L"] * (2.0 * math.pi * n1 * r1)
    C_c = theta["k_cap"] * c_wr_um**2 * 1e-15
    R_b = theta["k_res"] * r_l_um
    C_g = theta["k_cgs"] * w_tot_um * 1e-15

    j = torch.complex(torch.zeros_like(r1), torch.ones_like(r1))
    z_dev = R_b + 1.0 / (j * OMEGA * C_g)
    z_shunt = R_Lg + j * OMEGA * L_g
    z_mid = _series_parallel(z_dev, z_shunt)
    z_in = z_mid + 1.0 / (j * OMEGA * C_c)

    gamma = (z_in - Z0) / (z_in + Z0)
    return 20.0 * torch.log10(torch.clamp(gamma.abs(), min=1e-6))


def s22_db_model(x_raw: Tensor, theta: dict[str, Tensor]) -> Tensor:
    """S22 [dB] at f0 from the output network. x_raw: (..., 7) raw units."""
    r3, n3 = x_raw[..., 2], x_raw[..., 3]
    w_tot_um = x_raw[..., 6] * 1e6 * W_TOT_FACTOR

    L_d = _wheeler_L_H(r3, n3)
    R_Ld = theta["rho_L"] * (2.0 * math.pi * n3 * r3)
    C_d = theta["k_cds"] * w_tot_um * 1e-15
    r_o = theta["k_ro"] / torch.clamp(w_tot_um, min=1.0)

    j = torch.complex(torch.zeros_like(r3), torch.ones_like(r3))
    z_out = _series_parallel(
        R_Ld + j * OMEGA * L_d,
        1.0 / (j * OMEGA * C_d),
        r_o + 0.0 * j,
    )
    z_port = theta["k_bal"] * z_out
    gamma = (z_port - Z0) / (z_port + Z0)
    return 20.0 * torch.log10(torch.clamp(gamma.abs(), min=1e-6))


def _theta_tensors(log_theta: Tensor) -> dict[str, Tensor]:
    return {n: torch.exp(log_theta[i]) for i, n in enumerate(THETA_NAMES)}


def calibrate(
    x_raw: Tensor, s11_db: Tensor, s22_db: Tensor,
    n_steps: int = 2000, lr: float = 0.03, clip_db: float = -25.0,
) -> dict[str, float]:
    """Fit theta on a corpus by Huber loss over clipped dB values."""
    log_theta = torch.nn.Parameter(
        torch.log(torch.tensor(THETA_INIT, dtype=torch.double)))
    opt = torch.optim.Adam([log_theta], lr=lr)
    t11 = torch.clamp(s11_db, min=clip_db)
    t22 = torch.clamp(s22_db, min=clip_db)
    for _ in range(n_steps):
        opt.zero_grad()
        th = _theta_tensors(log_theta)
        p11 = torch.clamp(s11_db_model(x_raw, th), min=clip_db)
        p22 = torch.clamp(s22_db_model(x_raw, th), min=clip_db)
        loss = (torch.nn.functional.huber_loss(p11, t11, delta=3.0)
                + torch.nn.functional.huber_loss(p22, t22, delta=3.0))
        loss.backward()
        opt.step()
    theta = {n: float(v) for n, v in _theta_tensors(log_theta.detach()).items()}
    return theta


def save_theta(theta: dict[str, float]) -> None:
    THETA_PATH.parent.mkdir(exist_ok=True)
    json.dump(theta, open(THETA_PATH, "w"), indent=2)


def load_theta() -> dict[str, Tensor]:
    raw = json.load(open(THETA_PATH))
    return {n: torch.tensor(v, dtype=torch.double) for n, v in raw.items()}

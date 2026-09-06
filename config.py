"""PA Direct Geometry Optimization — Configuration (public release).

7D continuous BO: [L1L2_rad, L1L2_nr, L3L4_rad, L3L4_nr, C_wr, R_l, M_wr].

PDK-specific values (spiral trace width/spacing/metal layer/ground distance,
minimum inner diameter, MOS-cap ring counts, PDK/Spectre paths) are withheld
under the foundry NDA.  Copy ``pdk_private.example.py`` to ``pdk_private.py``
and fill in the values from your own PDK documentation to run simulations.
All analysis scripts run from the shipped logs in ``data/`` without a PDK.
"""

import numpy as np
from dataclasses import dataclass

try:
    from pdk_private import (  # noqa: F401
        IND_W, IND_SPACING, IND_LAY, IND_GDIS, D_IN_MIN_UM,
        CAP_BR, CAP_GR, PDK_INCLUDE, BALUN_VA, SPECTRE_BIN,
        REMOTE_HOST, REMOTE_DIR,
    )
except ImportError as _e:
    raise ImportError(
        "pdk_private.py not found. Copy pdk_private.example.py to "
        "pdk_private.py and fill in your PDK-specific values (withheld here "
        "under the foundry NDA). Analysis of the shipped data does not need it."
    ) from _e

# Fixed circuit parameters (public, see paper)
F0 = 2.4e9
VDD = 1.1
VBIAS = 0.5
Z0 = 50.0

# 7D search-space bounds (paper, Table II).  In the original workflow these
# were derived from a designer inductance spec via an NDA'd characterization
# table; the resulting bounds are public and hard-coded here.
L1L2_RAD_MIN, L1L2_RAD_MAX = 30.0, 45.0   # um
L1L2_NR_MIN,  L1L2_NR_MAX  = 4.5, 7.0
L3L4_RAD_MIN, L3L4_RAD_MAX = 30.0, 45.0
L3L4_NR_MIN,  L3L4_NR_MAX  = 3.5, 6.0
C_WR_MIN, C_WR_MAX = 1.1e-6, 3.2e-6       # m
R_L_MIN,  R_L_MAX  = 1.0e-6, 10.0e-6
M_WR_MIN, M_WR_MAX = 1.5e-6, 5.0e-6

N_DIMS = 7
DIM_NAMES = ["L1L2_rad", "L1L2_nr", "L3L4_rad", "L3L4_nr", "C_wr", "R_l", "M_wr"]
BOUNDS_LOWER = np.array([L1L2_RAD_MIN, L1L2_NR_MIN, L3L4_RAD_MIN, L3L4_NR_MIN,
                         C_WR_MIN, R_L_MIN, M_WR_MIN])
BOUNDS_UPPER = np.array([L1L2_RAD_MAX, L1L2_NR_MAX, L3L4_RAD_MAX, L3L4_NR_MAX,
                         C_WR_MAX, R_L_MAX, M_WR_MAX])


def check_d_in(rad_um: float, nr: float, w_um: float = None,
               sp_um: float = None) -> float:
    """Return inner diameter in um (must be > D_IN_MIN_UM)."""
    w = IND_W if w_um is None else w_um
    sp = IND_SPACING if sp_um is None else sp_um
    return 2 * rad_um - 2 * nr * (w + sp)


# Fixed component parameters for netlist (public)
RES_W = 1.0e-6
RES_MULTI = 12
MOS_LR = 40e-9
MOS_NR = 1
MOS_MULTI = 100

# Reconstructed-baseline design point (paper, Table II)
PRESIM = {
    "L1L2_rad": 35.0, "L1L2_nr": 6.5,
    "L3L4_rad": 45.0, "L3L4_nr": 4.5,
    "c_wr": 1.6e-6, "r_l": 1.55e-6, "m_wr": 3.1e-6,
}
PRESIM_X = np.array([PRESIM["L1L2_rad"], PRESIM["L1L2_nr"],
                     PRESIM["L3L4_rad"], PRESIM["L3L4_nr"],
                     PRESIM["c_wr"], PRESIM["r_l"], PRESIM["m_wr"]])
PRESIM_OBJECTIVES = {"PAE": 53.32, "Gain": 12.98, "Psat": 14.96}

# Constraint thresholds (public, see paper)
CONSTRAINTS = {
    "K_min": 1.0, "B1_min": 0.0, "S11_max_dB": -8.0, "S22_max_dB": -8.0,
    "OIP3_min_dBm": 18.0, "Pdc_max_mW": 25.0,
}

LOCAL_RESULTS_DIR = "results"


@dataclass(frozen=True)
class OptimizationConfig:
    n_initial: int = 20
    n_iterations: int = 40
    batch_size: int = 3
    seed: int = 42
    num_restarts: int = 10
    raw_samples: int = 512
    mc_samples: int = 128
    warmstart_frac: float = 0.5

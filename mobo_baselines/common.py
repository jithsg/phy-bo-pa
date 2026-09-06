"""Shared evaluation adapter for the four PA baseline optimizers.

Wraps the existing ``evaluate.evaluate_point`` so each baseline only needs to
handle its own optimizer-specific glue. The objective convention matches the
proposed BO loops:

  * Three objectives (all maximised internally; we expose them in maximise form
    and the per-baseline runner negates if its library wants minimisation):
        PAE_pct, Gain_dB, Psat_dBm
  * Six constraint margins (positive ⇒ feasible):
        c_K     = K        - K_min
        c_B1    = B1       - B1_min
        c_S11   = S11_max  - S11_dB
        c_S22   = S22_max  - S22_dB
        c_OIP3  = OIP3_dBm - OIP3_min
        c_Pdc   = Pdc_max  - Pdc_mW    (uses the same snap-margin as bo_loop_geo)

All snapping, d_in checks, netlist generation, and Spectre execution are
delegated to ``evaluate.evaluate_point``, so the proposed pipeline and the
baselines exercise an identical evaluation channel.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

# Make the PA project root importable when this package is invoked as
# ``python -m mobo_baselines.run_pymoo`` from inside the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BOUNDS_LOWER, BOUNDS_UPPER, CONSTRAINTS, DIM_NAMES, N_DIMS  # noqa: E402
from evaluate import evaluate_point  # noqa: E402

logger = logging.getLogger(__name__)

OBJ_NAMES: tuple[str, ...] = ("PAE_pct", "Gain_dB", "Psat_dBm")

# Same Pdc snap-margin as bo_loop_geo.py — keeps feasibility classification
# identical between baselines and the proposed flow.
PDC_SNAP_MARGIN_MW = 0.5


# ---------------------------------------------------------------
# Records
# ---------------------------------------------------------------

@dataclass
class BaselineEvalRecord:
    """One evaluation: design vector, raw metrics, constraint margins, feasibility."""

    eval_id: int
    x: list[float]
    # Raw objectives (None if simulation failed / rejected)
    pae_pct: Optional[float]
    gain_db: Optional[float]
    psat_dbm: Optional[float]
    # Raw constraint metrics (None if rejected)
    K: Optional[float]
    B1: Optional[float]
    S11_dB: Optional[float]
    S22_dB: Optional[float]
    OIP3_dBm: Optional[float]
    Pdc_mW: Optional[float]
    # Derived constraint margins (positive ⇒ feasible)
    c_K: float
    c_B1: float
    c_S11: float
    c_S22: float
    c_OIP3: float
    c_Pdc: float
    is_feasible: bool
    rejected: bool          # True if pre-sim d_in rejection
    sim_time_s: float
    point_id: str


@dataclass
class BaselineTrajectory:
    """Full trajectory for one baseline + seed."""

    baseline_name: str
    seed: int
    n_evals: int
    ref_point_obj: list[float]     # reference point in negated/min form for HV
    records: list[BaselineEvalRecord] = field(default_factory=list)
    hv_history: list[float] = field(default_factory=list)
    feasible_count: int = 0
    final_hv: float = 0.0


# ---------------------------------------------------------------
# Bounds / parameter helpers
# ---------------------------------------------------------------

def bounds_array() -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) as 1-D numpy arrays, length N_DIMS."""
    return np.asarray(BOUNDS_LOWER, dtype=float), np.asarray(BOUNDS_UPPER, dtype=float)


def normalize(x: np.ndarray) -> np.ndarray:
    """Map x from original SI units to [0, 1]^N_DIMS."""
    lo, hi = bounds_array()
    return (x - lo) / (hi - lo)


def unnormalize(u: np.ndarray) -> np.ndarray:
    """Map u from [0, 1]^N_DIMS back to original SI units."""
    lo, hi = bounds_array()
    return lo + u * (hi - lo)


# ---------------------------------------------------------------
# Constraint thresholds (mirror bo_loop_geo conventions)
# ---------------------------------------------------------------

def constraint_margins(constraints: dict) -> dict:
    """Return positive-is-feasible margins for the six PA constraints."""
    return {
        "c_K":    constraints["K"]        - CONSTRAINTS["K_min"],
        "c_B1":   constraints["B1"]       - CONSTRAINTS["B1_min"],
        "c_S11":  CONSTRAINTS["S11_max_dB"] - constraints["S11_dB"],
        "c_S22":  CONSTRAINTS["S22_max_dB"] - constraints["S22_dB"],
        "c_OIP3": constraints["OIP3_dBm"] - CONSTRAINTS["OIP3_min_dBm"],
        "c_Pdc":  (CONSTRAINTS["Pdc_max_mW"] - PDC_SNAP_MARGIN_MW) - constraints["Pdc_mW"],
    }


def is_feasible(margins: dict) -> bool:
    return all(v >= 0 for v in margins.values())


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def evaluate_design(x: np.ndarray, eval_id: int, baseline_name: str, seed: int,
                    simulator: str = "spectre") -> BaselineEvalRecord:
    """Evaluate one design point and package the result.

    Args:
        x: (7,) design vector in original SI units (as defined by BOUNDS_*).
        eval_id: zero-based evaluation index for this run.
        baseline_name: short name used in the point_id (e.g. "nsga2").
        seed: campaign seed; included in point_id for uniqueness.
        simulator: "spectre" runs real Spectre via ``evaluate_point``;
                   "mock" uses ``mock_pa_evaluate`` for fast pipeline tests.
    """
    point_id = f"baselines_{baseline_name}_s{seed}_eval{eval_id:04d}"

    if simulator == "mock":
        result = mock_pa_evaluate(x)
    elif simulator == "spectre":
        result = evaluate_point(x, point_id)
    else:
        raise ValueError(f"unknown simulator {simulator!r}")

    rejected = bool(result.get("rejected", False))
    constraints = result.get("constraints", {})
    objectives = result.get("objectives", {})

    margins = constraint_margins(constraints)
    feas = (not rejected) and is_feasible(margins)

    return BaselineEvalRecord(
        eval_id=eval_id,
        x=list(map(float, x.tolist())),
        pae_pct=objectives.get("PAE_pct"),
        gain_db=objectives.get("Gain_dB"),
        psat_dbm=objectives.get("Psat_dBm"),
        K=constraints.get("K"),
        B1=constraints.get("B1"),
        S11_dB=constraints.get("S11_dB"),
        S22_dB=constraints.get("S22_dB"),
        OIP3_dBm=constraints.get("OIP3_dBm"),
        Pdc_mW=constraints.get("Pdc_mW"),
        **margins,
        is_feasible=feas,
        rejected=rejected,
        sim_time_s=float(result.get("sim_time_s", 0.0)),
        point_id=point_id,
    )


# ---------------------------------------------------------------
# Mock simulator (for fast end-to-end testing without Spectre)
# ---------------------------------------------------------------

def mock_pa_evaluate(x: np.ndarray) -> dict:
    """Synthetic PA evaluator. Deterministic in x, no Spectre needed.

    Uses smooth analytical surrogates of the form expected for a Class-AB / -E
    push-pull PA at 2.4 GHz: PAE rises with current, Gain peaks at moderate
    matching, Psat scales with M_wr, with realistic constraint trade-offs.
    Intended only for pipeline validation — NOT a calibrated model.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = bounds_array()
    u = (x - lo) / (hi - lo)  # in [0, 1] per dim

    # Crude objective surrogates (smooth in u, peaked-or-saturating in the
    # right qualitative direction for a push-pull PA).
    pae = 35.0 * (1 - (u[6] - 0.55) ** 2) * (1 - 0.5 * (u[4] - 0.45) ** 2)
    gain = 22.0 + 4.0 * u[2] - 3.0 * (u[6] - 0.4) ** 2
    psat = 18.0 + 6.0 * u[6] + 2.0 * u[2]

    # Constraint surrogates — engineered to be marginal at PRESIM-like inputs.
    K = 1.05 + 0.5 * u[5] - 0.4 * u[6]
    B1 = 0.3 - 0.5 * (u[6] - 0.6) + 0.2 * u[3]
    S11_dB = -10.0 + 4.0 * (u[0] - 0.5) ** 2 + 3.0 * (u[1] - 0.5) ** 2
    S22_dB = -10.0 + 4.0 * (u[2] - 0.5) ** 2 + 3.0 * (u[3] - 0.5) ** 2
    OIP3 = 17.0 + 8.0 * u[6] + 2.0 * u[4]
    Pdc = 22.0 + 8.0 * u[6] - 2.0 * u[5]

    return {
        "objectives": {"PAE_pct": pae, "Gain_dB": gain, "Psat_dBm": psat},
        "constraints": {
            "K": K, "B1": B1, "S11_dB": S11_dB, "S22_dB": S22_dB,
            "OIP3_dBm": OIP3, "Pdc_mW": Pdc,
        },
        "rejected": False,
        "sim_time_s": 0.0,
    }


# ---------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------

def feasible_hypervolume(
    records: list[BaselineEvalRecord], ref_point_obj: list[float]
) -> float:
    """Hypervolume of the *feasible* points in (PAE, Gain, Psat) space.

    All three objectives are maximised. BoTorch's FastNondominatedPartitioning
    also maximises, so we pass objectives un-negated and use a reference point
    that is dominated by all Pareto-optimal points (i.e. a lower bound).
    """
    feasible = [r for r in records if r.is_feasible]
    if not feasible:
        return 0.0
    Y = torch.tensor(
        [[r.pae_pct, r.gain_db, r.psat_dbm] for r in feasible],
        dtype=torch.double,
    )
    ref = torch.tensor(ref_point_obj, dtype=torch.double)
    return float(FastNondominatedPartitioning(ref_point=ref, Y=Y)
                 .compute_hypervolume().item())


def default_ref_point() -> list[float]:
    """Reference point in (PAE, Gain, Psat) space, in the maximisation sense.

    Set to a worst-acceptable lower bound per objective so the HV ordering
    across runs is comparable:
      * PAE  >= 0%
      * Gain >= 0 dB
      * Psat >= 0 dBm
    """
    return [0.0, 0.0, 0.0]


# ---------------------------------------------------------------
# Stateful evaluator factory
# ---------------------------------------------------------------

def make_eval_fn(
    trajectory: BaselineTrajectory,
    simulator: str = "spectre",
) -> Callable[[np.ndarray], BaselineEvalRecord]:
    """Build a stateful evaluator that appends to the trajectory in-place.

    The returned callable takes a (7,) design vector and returns the eval
    record. HV is recomputed after each eval so the trajectory is always
    plot-ready.
    """

    def _eval(x: np.ndarray) -> BaselineEvalRecord:
        eval_id = len(trajectory.records)
        record = evaluate_design(
            x=x,
            eval_id=eval_id,
            baseline_name=trajectory.baseline_name,
            seed=trajectory.seed,
            simulator=simulator,
        )
        trajectory.records.append(record)
        if record.is_feasible:
            trajectory.feasible_count += 1
        hv = feasible_hypervolume(trajectory.records, trajectory.ref_point_obj)
        trajectory.hv_history.append(hv)
        trajectory.final_hv = hv

        status = (
            "REJECT" if record.rejected
            else ("FEAS" if record.is_feasible else "INFEAS")
        )
        if record.pae_pct is not None:
            logger.info(
                "  eval %4d [%s] PAE=%.2f%% Gain=%.2fdB Psat=%.2fdBm "
                "OIP3=%.1fdBm Pdc=%.2fmW S11=%.2fdB HV=%.4f",
                eval_id, status,
                record.pae_pct, record.gain_db, record.psat_dbm,
                record.OIP3_dBm, record.Pdc_mW, record.S11_dB, hv,
            )
        else:
            logger.info("  eval %4d [%s] (no metrics) HV=%.4f", eval_id, status, hv)
        return record

    return _eval


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

def save_trajectory(trajectory: BaselineTrajectory, output_dir: Path) -> Path:
    """Write trajectory JSON to results/mobo_baselines/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{trajectory.baseline_name}_seed{trajectory.seed}.json"
    payload = {
        "baseline_name": trajectory.baseline_name,
        "seed": trajectory.seed,
        "n_evals": trajectory.n_evals,
        "ref_point_obj": trajectory.ref_point_obj,
        "feasible_count": trajectory.feasible_count,
        "final_hv": trajectory.final_hv,
        "hv_history": trajectory.hv_history,
        "records": [asdict(r) for r in trajectory.records],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved trajectory to %s", out_path)
    return out_path


# ---------------------------------------------------------------
# Misc
# ---------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def add_simulator_args(parser) -> None:
    """Add the standard --simulator switch to a runner's argparse parser."""
    parser.add_argument(
        "--simulator", choices=["mock", "spectre"], default="mock",
        help="'mock' uses the synthetic PA model (no SSH/Spectre); "
             "'spectre' runs real Spectre via evaluate.evaluate_point.",
    )

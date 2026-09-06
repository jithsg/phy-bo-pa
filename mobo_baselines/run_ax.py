"""Ax (Meta) qNEHVI baseline for the PA problem.

3 objectives + 6 outcome constraints, all defined natively in Ax's Service
API. Parameters are passed in normalised [0,1] space and unnormalised inside
the eval callback (Ax rejects raw SI-scale ranges as numerically unstable;
this mirrors the VCO setup).

Run:
    PYTHONPATH=. \\
    ~/.venvs/vco-baselines/bin/python -m mobo_baselines.run_ax \\
        --simulator spectre --n-evals 180 --seed 42
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ax")

from ax.service.ax_client import AxClient  # noqa: E402
from ax.service.utils.instantiation import ObjectiveProperties  # noqa: E402

from .common import (  # noqa: E402
    BaselineTrajectory,
    add_simulator_args,
    bounds_array,
    default_ref_point,
    make_eval_fn,
    save_trajectory,
    seed_everything,
    setup_logging,
)

logger = logging.getLogger(__name__)

BASELINE_NAME = "ax_mobo"

DIM_NAMES = (
    "L1L2_rad", "L1L2_nr", "L3L4_rad", "L3L4_nr", "C_wr", "R_l", "M_wr",
)


def run_ax(
    n_evals: int = 180,
    seed: int = 42,
    simulator: str = "spectre",
    output_dir: Path = Path("results/mobo_baselines"),
) -> BaselineTrajectory:
    setup_logging()
    seed_everything(seed)

    lo, hi = bounds_array()

    trajectory = BaselineTrajectory(
        baseline_name=BASELINE_NAME,
        seed=seed,
        n_evals=n_evals,
        ref_point_obj=default_ref_point(),
    )
    eval_fn = make_eval_fn(trajectory, simulator=simulator)

    # Ax rejects raw SI ranges (e.g. 1e-6 widths) → normalise to [0,1].
    parameters = [
        {"name": name, "type": "range", "bounds": [0.0, 1.0], "value_type": "float"}
        for name in DIM_NAMES
    ]

    client = AxClient(random_seed=seed, verbose_logging=False)
    client.create_experiment(
        name="pa_baseline_ax",
        parameters=parameters,
        objectives={
            # All three maximised. Thresholds match common.default_ref_point().
            "PAE_pct":  ObjectiveProperties(minimize=False, threshold=0.0),
            "Gain_dB":  ObjectiveProperties(minimize=False, threshold=0.0),
            "Psat_dBm": ObjectiveProperties(minimize=False, threshold=0.0),
        },
        # Ax outcome_constraints are strings in the form "metric op threshold".
        # Internally Ax stores these as metric - threshold and requires
        # constraint <= 0; we feed the raw measured constraint metric and
        # use the same thresholds as common.CONSTRAINTS / bo_loop_geo.
        outcome_constraints=[
            "K >= 1.0",
            "B1 >= 0.0",
            "S11_dB <= -8.0",
            "S22_dB <= -8.0",
            "OIP3_dBm >= 18.0",
            "Pdc_mW <= 24.5",          # includes 0.5 mW snap margin
        ],
    )

    logger.info("=" * 64)
    logger.info("Ax MOBO (qNEHVI) | budget=%d | seed=%d | sim=%s",
                n_evals, seed, simulator)
    logger.info("=" * 64)

    for _ in range(n_evals):
        params, trial_idx = client.get_next_trial()
        x_norm = np.array([float(params[name]) for name in DIM_NAMES])
        x_vec = lo + x_norm * (hi - lo)
        record = eval_fn(x_vec)

        if record.pae_pct is None or record.rejected:
            raw_data = {
                "PAE_pct":  (0.0, 0.0),  "Gain_dB":  (0.0, 0.0),
                "Psat_dBm": (0.0, 0.0),
                "K":        (0.0, 0.0),  "B1":       (-1.0, 0.0),
                "S11_dB":   (0.0, 0.0),  "S22_dB":   (0.0, 0.0),
                "OIP3_dBm": (0.0, 0.0),  "Pdc_mW":   (999.0, 0.0),
            }
        else:
            raw_data = {
                "PAE_pct":  (record.pae_pct, 0.0),
                "Gain_dB":  (record.gain_db, 0.0),
                "Psat_dBm": (record.psat_dbm, 0.0),
                "K":        (record.K, 0.0),
                "B1":       (record.B1, 0.0),
                "S11_dB":   (record.S11_dB, 0.0),
                "S22_dB":   (record.S22_dB, 0.0),
                "OIP3_dBm": (record.OIP3_dBm, 0.0),
                "Pdc_mW":   (record.Pdc_mW, 0.0),
            }
        client.complete_trial(trial_index=trial_idx, raw_data=raw_data)

    logger.info("Done. evals=%d feasible=%d final_HV=%.4f",
                len(trajectory.records), trajectory.feasible_count,
                trajectory.final_hv)
    save_trajectory(trajectory, output_dir)
    return trajectory


def main():
    p = argparse.ArgumentParser(description="Ax MOBO baseline (PA)")
    p.add_argument("--n-evals", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("results/mobo_baselines"),
    )
    add_simulator_args(p)
    args = p.parse_args()

    run_ax(
        n_evals=args.n_evals,
        seed=args.seed,
        simulator=args.simulator,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

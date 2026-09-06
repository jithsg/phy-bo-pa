"""MOTPE baseline via Optuna for the PA problem.

3 objectives (PAE, Gain, Psat — all minimised internally by Optuna; we pass
negated so that minimising = maximising the originals) + 6 constraints via
Optuna's ``constraints_func`` API (each value <= 0 is feasible; we feed
``-c_i`` so c_i >= 0 from common.py maps directly).

Run:
    PYTHONPATH=. \\
    ~/.venvs/vco-baselines/bin/python -m mobo_baselines.run_optuna \\
        --simulator spectre --n-evals 180 --seed 42
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

from .common import (
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

BASELINE_NAME = "optuna_motpe"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    return trial.user_attrs["constraints"]


def run_motpe(
    n_evals: int = 180,
    seed: int = 42,
    n_startup: int = 20,
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

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        x = np.array([
            trial.suggest_float("L1L2_rad", lo[0], hi[0]),
            trial.suggest_float("L1L2_nr",  lo[1], hi[1]),
            trial.suggest_float("L3L4_rad", lo[2], hi[2]),
            trial.suggest_float("L3L4_nr",  lo[3], hi[3]),
            trial.suggest_float("C_wr",     lo[4], hi[4]),
            trial.suggest_float("R_l",      lo[5], hi[5]),
            trial.suggest_float("M_wr",     lo[6], hi[6]),
        ])
        record = eval_fn(x)

        if record.pae_pct is None or record.rejected:
            trial.set_user_attr("constraints", (1e3,) * 6)
            return 0.0, 0.0, 0.0

        # Optuna constraint convention: c <= 0 feasible.
        # common.py gives c >= 0 feasible → negate.
        trial.set_user_attr("constraints", (
            -record.c_K, -record.c_B1, -record.c_S11,
            -record.c_S22, -record.c_OIP3, -record.c_Pdc,
        ))
        # Optuna minimises → negate the maximised objectives
        return -record.pae_pct, -record.gain_db, -record.psat_dbm

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=n_startup,
        multivariate=True,
        group=True,
        constraints_func=_constraints,
    )
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    logger.info("=" * 64)
    logger.info("MOTPE (Optuna) | budget=%d | n_startup=%d | seed=%d | sim=%s",
                n_evals, n_startup, seed, simulator)
    logger.info("=" * 64)

    study.optimize(objective, n_trials=n_evals, show_progress_bar=False)

    logger.info("Done. evals=%d feasible=%d final_HV=%.4f",
                len(trajectory.records), trajectory.feasible_count,
                trajectory.final_hv)
    save_trajectory(trajectory, output_dir)
    return trajectory


def main():
    p = argparse.ArgumentParser(description="MOTPE baseline (PA)")
    p.add_argument("--n-evals", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-startup", type=int, default=20)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("results/mobo_baselines"),
    )
    add_simulator_args(p)
    args = p.parse_args()

    run_motpe(
        n_evals=args.n_evals,
        seed=args.seed,
        n_startup=args.n_startup,
        simulator=args.simulator,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

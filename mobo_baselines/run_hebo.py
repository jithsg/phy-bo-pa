"""HEBO baseline (GeneralBO) for the PA problem.

3 objectives + 6 constraints. HEBO's GeneralBO accepts both natively. Same
ref_point=None workaround we used in the VCO project so the constrained MO
suggest path doesn't trip the ref_point assertion.

Run:
    PYTHONPATH=. \\
    ~/.venvs/vco-baselines/bin/python -m mobo_baselines.run_hebo \\
        --simulator spectre --n-evals 180 --seed 42
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO

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

BASELINE_NAME = "hebo_general"

DIM_NAMES = (
    "L1L2_rad", "L1L2_nr", "L3L4_rad", "L3L4_nr", "C_wr", "R_l", "M_wr",
)


def _build_design_space() -> DesignSpace:
    lo, hi = bounds_array()
    spec = [
        {"name": name, "type": "num", "lb": float(lo[i]), "ub": float(hi[i])}
        for i, name in enumerate(DIM_NAMES)
    ]
    return DesignSpace().parse(spec)


def run_hebo(
    n_evals: int = 180,
    seed: int = 42,
    rand_sample: int = 20,
    simulator: str = "spectre",
    output_dir: Path = Path("results/mobo_baselines"),
) -> BaselineTrajectory:
    setup_logging()
    seed_everything(seed)

    trajectory = BaselineTrajectory(
        baseline_name=BASELINE_NAME,
        seed=seed,
        n_evals=n_evals,
        ref_point_obj=default_ref_point(),
    )
    eval_fn = make_eval_fn(trajectory, simulator=simulator)

    space = _build_design_space()
    # GeneralBO.suggest asserts num_constr == 0 when ref_point is set
    # (constrained MO HV-EI selection is not implemented). Falling back to
    # ref_point=None keeps the constrained MO path enabled with uncertainty
    # selection — same workaround we used for the VCO project.
    opt = GeneralBO(
        space=space,
        num_obj=3,
        num_constr=6,
        rand_sample=rand_sample,
        ref_point=None,
    )

    logger.info("=" * 64)
    logger.info("HEBO GeneralBO | budget=%d | rand_sample=%d | seed=%d | sim=%s",
                n_evals, rand_sample, seed, simulator)
    logger.info("=" * 64)

    for _ in range(n_evals):
        x_df: pd.DataFrame = opt.suggest(n_suggestions=1)
        x_vec = np.array([float(x_df.iloc[0][name]) for name in DIM_NAMES])
        record = eval_fn(x_vec)

        if record.pae_pct is None or record.rejected:
            y = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])
        else:
            # HEBO minimises objectives and treats c_i <= 0 as feasible →
            # negate both the maximised objectives and the c >= 0 margins.
            y = np.array([[
                -record.pae_pct, -record.gain_db, -record.psat_dbm,
                -record.c_K, -record.c_B1, -record.c_S11,
                -record.c_S22, -record.c_OIP3, -record.c_Pdc,
            ]])
        opt.observe(x_df, y)

    logger.info("Done. evals=%d feasible=%d final_HV=%.4f",
                len(trajectory.records), trajectory.feasible_count,
                trajectory.final_hv)
    save_trajectory(trajectory, output_dir)
    return trajectory


def main():
    p = argparse.ArgumentParser(description="HEBO baseline (PA)")
    p.add_argument("--n-evals", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rand-sample", type=int, default=20)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("results/mobo_baselines"),
    )
    add_simulator_args(p)
    args = p.parse_args()

    run_hebo(
        n_evals=args.n_evals,
        seed=args.seed,
        rand_sample=args.rand_sample,
        simulator=args.simulator,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

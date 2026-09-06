"""NSGA-II baseline via pymoo for the PA optimization problem.

3 objectives (PAE, Gain, Psat — all maximised) + 6 inequality constraints
(c >= 0 feasible from common.py; pymoo's convention is g <= 0, so we negate).

Run:
    PYTHONPATH=. \\
    ~/.venvs/vco-baselines/bin/python -m mobo_baselines.run_pymoo \\
        --simulator spectre --n-evals 180 --seed 42
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination

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

BASELINE_NAME = "pymoo_nsga2"


class PAProblem(ElementwiseProblem):
    """pymoo problem definition: 7 vars, 3 objectives (negated → minimised),
    6 inequality constraints (negated → g <= 0 feasible)."""

    # Penalty values for rejected / failed simulations
    PENALTY_OBJ = np.array([0.0, 0.0, 0.0])     # 0 PAE / 0 Gain / 0 Psat (negated → 0)
    PENALTY_CON = np.array([1e3] * 6)            # large positive g ⇒ violated

    def __init__(self, eval_fn):
        lo, hi = bounds_array()
        super().__init__(n_var=7, n_obj=3, n_ieq_constr=6, xl=lo, xu=hi)
        self._eval_fn = eval_fn

    def _evaluate(self, x, out, *args, **kwargs):
        record = self._eval_fn(np.asarray(x, dtype=float))

        if record.pae_pct is None or record.rejected:
            out["F"] = self.PENALTY_OBJ.copy()
            out["G"] = self.PENALTY_CON.copy()
            return

        # pymoo minimises → negate the (maximised) objectives
        out["F"] = np.array([-record.pae_pct, -record.gain_db, -record.psat_dbm])
        # pymoo wants g <= 0 feasible; common.py gives c >= 0 feasible → negate
        out["G"] = np.array([
            -record.c_K, -record.c_B1, -record.c_S11,
            -record.c_S22, -record.c_OIP3, -record.c_Pdc,
        ])


def run_nsga2(
    n_evals: int = 180,
    seed: int = 42,
    pop_size: int = 20,
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
    problem = PAProblem(eval_fn=eval_fn)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    termination = MaximumFunctionCallTermination(n_max_evals=n_evals)

    logger.info("=" * 64)
    logger.info("NSGA-II (pymoo) | budget=%d | pop=%d | seed=%d | sim=%s",
                n_evals, pop_size, seed, simulator)
    logger.info("=" * 64)

    minimize(problem, algorithm, termination, seed=seed, verbose=False)

    # pymoo can overshoot by a few evals on the final batch — truncate.
    if len(trajectory.records) > n_evals:
        trajectory.records = trajectory.records[:n_evals]
        trajectory.hv_history = trajectory.hv_history[:n_evals]
        trajectory.feasible_count = sum(r.is_feasible for r in trajectory.records)
        trajectory.final_hv = trajectory.hv_history[-1]

    logger.info("Done. evals=%d feasible=%d final_HV=%.4f",
                len(trajectory.records), trajectory.feasible_count,
                trajectory.final_hv)
    save_trajectory(trajectory, output_dir)
    return trajectory


def main():
    p = argparse.ArgumentParser(description="NSGA-II baseline (PA)")
    p.add_argument("--n-evals", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pop-size", type=int, default=20)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("results/mobo_baselines"),
    )
    add_simulator_args(p)
    args = p.parse_args()

    run_nsga2(
        n_evals=args.n_evals,
        seed=args.seed,
        pop_size=args.pop_size,
        simulator=args.simulator,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

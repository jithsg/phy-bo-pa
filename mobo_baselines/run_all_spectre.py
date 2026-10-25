"""Sequentially run all 4 PA baselines on real Spectre, then aggregate.

Designed for unattended overnight execution. Captures per-baseline runtime,
catches per-baseline failures so one bad run does not kill the campaign,
and writes a final comparison alongside the existing geo2/geo21 results.

Run:
    PYTHONPATH=. \\
    nohup ~/.venvs/vco-baselines/bin/python -m mobo_baselines.run_all_spectre \\
        --n-evals 180 --seed 42 \\
        > results/mobo_baselines/spectre_campaign.log 2>&1 &
"""

import argparse
import logging
import time
import traceback
from pathlib import Path

from . import compare, run_ax, run_hebo, run_optuna, run_pymoo
from .common import setup_logging

logger = logging.getLogger(__name__)

BASELINES = [
    ("pymoo_nsga2", run_pymoo.run_nsga2, {}),
    ("optuna_motpe", run_optuna.run_motpe, {}),
    ("hebo_general", run_hebo.run_hebo, {}),
    ("ax_mobo", run_ax.run_ax, {}),
    ("pymoo_nsga3", run_pymoo.run_nsga3, {}),
    ("pymoo_moead", run_pymoo.run_moead, {}),
]


def main():
    parser = argparse.ArgumentParser(description="Run all 4 PA baselines on Spectre")
    parser.add_argument("--n-evals", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/mobo_baselines"),
    )
    parser.add_argument(
        "--only", nargs="*", choices=[b[0] for b in BASELINES], default=None,
        help="Restrict to a subset of baselines.",
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Skip the final aggregation (used by worker lanes; the master aggregates).",
    )
    args = parser.parse_args()

    setup_logging()
    selected = [(n, fn, k) for (n, fn, k) in BASELINES if not args.only or n in args.only]
    logger.info("Campaign: %d baselines | n_evals=%d | seed=%d",
                len(selected), args.n_evals, args.seed)

    timings: dict[str, float] = {}
    failures: dict[str, str] = {}

    for name, run_fn, extra in selected:
        logger.info("\n" + "=" * 70)
        logger.info(">>> %s (Spectre)", name)
        logger.info("=" * 70)

        t0 = time.monotonic()
        try:
            run_fn(
                n_evals=args.n_evals,
                seed=args.seed,
                output_dir=args.output_dir,
                simulator="spectre",
                **extra,
            )
            timings[name] = time.monotonic() - t0
            logger.info(">>> %s done in %.1f min", name, timings[name] / 60)
        except Exception as e:
            timings[name] = time.monotonic() - t0
            failures[name] = f"{type(e).__name__}: {e}"
            logger.error(">>> %s FAILED after %.1f min: %s",
                         name, timings[name] / 60, e)
            logger.error(traceback.format_exc())

    logger.info("\n" + "=" * 70)
    logger.info("Campaign timings:")
    for name, secs in timings.items():
        marker = "FAIL" if name in failures else "ok"
        logger.info("  %-15s  %6.1f min  [%s]", name, secs / 60, marker)
    if failures:
        logger.warning("Failures:")
        for name, err in failures.items():
            logger.warning("  %s: %s", name, err)

    if args.no_compare:
        logger.info("Skipping aggregation (--no-compare).")
        return
    logger.info("\nAggregating comparison...")
    out_dir = args.output_dir / "comparison_spectre"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare.write_comparison(args.output_dir, out_dir)


if __name__ == "__main__":
    main()

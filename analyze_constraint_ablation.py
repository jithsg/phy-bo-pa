"""Per-seed report for the matched constraint-only ablation (REMAINING_WORK item 1).

Arm A (no d_in inequality): results/cabl_nodin_s{seed}
Arm B (with inequality)   : results/cabl_din_s42 for seed 42,
                            results/ms_stage1_s{seed} for seeds 7/11/23/101
                            (same code path, bit-identical DoE; verified by
                            campaign_constraint_ablation.py --check).

Reports, per seed and pooled: invalid-geometry (d_in-rejected) evaluations,
BO-phase and total feasible designs, best feasible PAE, final hypervolume over
feasible designs under a per-seed reference frozen over the union of both arms,
Spectre failures, and total simulation time.  Prints a LaTeX table body and
writes results/constraint_ablation_summary.csv.

Usage: python3 analyze_constraint_ablation.py [--partial]
  --partial: report arms that are still running (n < 100) instead of skipping.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from scipy.stats import fisher_exact

BASE = Path(__file__).parent
RESULTS = BASE / "results"
SEEDS = [7, 11, 23, 101, 42]
OBJ = ["PAE_pct", "Gain_dB", "Psat_dBm"]
N_TOTAL = 100
N_BO = 80


def _load(d: Path) -> list[dict] | None:
    p = d / "all_results.json"
    return json.load(open(p)) if p.exists() else None


def _arm_dirs(seed: int) -> tuple[Path, Path]:
    arm_a = RESULTS / f"cabl_nodin_s{seed}"
    arm_b = RESULTS / ("cabl_din_s42" if seed == 42 else f"ms_stage1_s{seed}")
    return arm_a, arm_b


def _objs(recs: list[dict]) -> np.ndarray:
    return np.array([[r["objectives"][k] for k in OBJ] for r in recs])


def _feas(recs: list[dict]) -> np.ndarray:
    return np.array([bool(r.get("feasible")) for r in recs])


def _hv(Y: np.ndarray, ref: np.ndarray) -> float:
    if len(Y) == 0:
        return 0.0
    part = FastNondominatedPartitioning(
        ref_point=torch.tensor(ref, dtype=torch.double),
        Y=torch.tensor(Y, dtype=torch.double),
    )
    return part.compute_hypervolume().item()


def _ref(Y: np.ndarray) -> np.ndarray:
    lo, hi = Y.min(0), Y.max(0)
    return lo - 0.1 * np.maximum(hi - lo, 1e-6)


def _stats(recs: list[dict], ref: np.ndarray) -> dict:
    Y, f = _objs(recs), _feas(recs)
    bo = [r for r in recs if "doe" not in r["point_id"]]
    return {
        "n": len(recs),
        "invalid": sum(1 for r in recs if r.get("rejected")),
        "sim_fail": sum(1 for r in recs if r.get("errors") and not r.get("rejected")),
        "feas_bo": sum(1 for r in bo if r.get("feasible")),
        "feas_tot": int(f.sum()),
        "best_pae": float(Y[f][:, 0].max()) if f.any() else float("nan"),
        "hv": _hv(Y[f], ref),
        "sim_time_min": sum(r.get("sim_time_s", 0.0) for r in recs) / 60.0,
    }


def main(argv: list[str]) -> None:
    partial = "--partial" in argv
    rows, pooled = [], {"A": dict(invalid=0, feas_bo=0, n_bo=0), "B": dict(invalid=0, feas_bo=0, n_bo=0)}
    print(f"{'seed':>4} | {'arm':3} | {'n':>3} | {'invalid':>7} | {'feasBO':>6} | {'feasTot':>7} | "
          f"{'bestPAE':>7} | {'HV':>8} | {'simfail':>7} | {'sim_min':>7}")
    for s in SEEDS:
        da, db = _arm_dirs(s)
        ra, rb = _load(da), _load(db)
        if ra is None or rb is None:
            print(f"{s:>4} | missing arm ({'A' if ra is None else 'B'})")
            continue
        if (len(ra) < N_TOTAL or len(rb) < N_TOTAL) and not partial:
            print(f"{s:>4} | incomplete (A n={len(ra)}, B n={len(rb)}) -- use --partial")
            continue
        n = min(len(ra), len(rb), N_TOTAL)
        ra, rb = ra[:n], rb[:n]
        ref = _ref(np.vstack([_objs(ra), _objs(rb)]))
        for arm, recs in (("A", ra), ("B", rb)):
            st = _stats(recs, ref)
            n_bo = max(0, len(recs) - 20)
            pooled[arm]["invalid"] += st["invalid"]
            pooled[arm]["feas_bo"] += st["feas_bo"]
            pooled[arm]["n_bo"] += n_bo
            print(f"{s:>4} | {arm:3} | {st['n']:>3} | {st['invalid']:>7} | {st['feas_bo']:>6} | "
                  f"{st['feas_tot']:>7} | {st['best_pae']:>7.1f} | {st['hv']:>8.0f} | "
                  f"{st['sim_fail']:>7} | {st['sim_time_min']:>7.1f}")
            rows.append({"seed": s, "arm": arm, **st})

    if rows:
        out = RESULTS / "constraint_ablation_summary.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")

    a, b = pooled["A"], pooled["B"]
    if a["n_bo"] and b["n_bo"]:
        p_inv = fisher_exact([[a["invalid"], a["n_bo"] - a["invalid"]],
                              [b["invalid"], b["n_bo"] - b["invalid"]]])[1]
        p_feas = fisher_exact([[a["feas_bo"], a["n_bo"] - a["feas_bo"]],
                               [b["feas_bo"], b["n_bo"] - b["feas_bo"]]])[1]
        print(f"\nPooled BO-phase: invalid A {a['invalid']}/{a['n_bo']} vs B {b['invalid']}/{b['n_bo']} "
              f"(Fisher two-sided p={p_inv:.3g}); feasible A {a['feas_bo']}/{a['n_bo']} vs "
              f"B {b['feas_bo']}/{b['n_bo']} (p={p_feas:.3g})")

    # LaTeX rows: seed & invalid A/B & feas A/B & bestPAE A/B & HV A/B
    by_seed: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r
    print("\n% LaTeX rows (no-d_in / with-d_in):")
    for s in SEEDS:
        if s not in by_seed or len(by_seed[s]) < 2:
            continue
        A, B = by_seed[s]["A"], by_seed[s]["B"]
        print(f"{s} & {A['invalid']} / {B['invalid']} & {A['feas_bo']} / {B['feas_bo']} & "
              f"{A['best_pae']:.1f} / {B['best_pae']:.1f} & {A['hv']:,.0f} / {B['hv']:,.0f} \\\\"
              .replace(",", "{,}"))


if __name__ == "__main__":
    main(sys.argv[1:])

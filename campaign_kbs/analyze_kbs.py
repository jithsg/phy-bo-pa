"""Analysis of the KBS-revision campaign: Stage-I budget sweep and external
baselines, five seeds, with the paper's hypervolume convention (feasible designs
only; reference point = min - 0.1*range over the union of the runs compared,
held fixed within each comparison).

    python3 campaign_kbs/analyze_kbs.py sweep      # N1 sweep table (+ LaTeX rows)
    python3 campaign_kbs/analyze_kbs.py baselines  # external optimizers vs ours
    python3 campaign_kbs/analyze_kbs.py all

Data sources (results/):
  sweep branches   kbs_results/br_n{N1}_s{S}/sw_branch_n{N1}_s{S}/all_results.json
  Stage-I ext      kbs_results/ext_s{S}/sw_stage1ext_s{S}/all_results.json   (N1=160)
  paper two-stage  ms_stage2_s{S}/all_results.json, geo21_warm_geoaware_s42 (N1=80)
  baselines        kbs_results/bl_{m}_s{S}/mobo_baselines/{m}_seed{S}.json and
                   mobo_baselines/{m}_seed42.json (May 2026 run)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "data" / "runs"   # public layout: data/runs/<campaign>
KBS = RES / "kbs_results"
SEEDS = (42, 7, 11, 23, 101)
N1_LIST = (0, 20, 40, 60, 80, 100, 120, 160)
BASELINES = ("hebo_general", "ax_mobo", "optuna_motpe", "pymoo_nsga2", "pymoo_nsga3", "pymoo_moead")
OBJ = ("PAE_pct", "Gain_dB", "Psat_dBm")
BUDGET = 180
N_DOE = 20


# ----------------------------------------------------------------------------- io
def _load(path: Path) -> list[dict] | None:
    return json.load(open(path)) if path.exists() else None


def sweep_run(seed: int, n1: int) -> list[dict] | None:
    """All results (DoE + BO) of the sweep condition with N1 zero-mean iterations."""
    if n1 == 80:
        own = KBS / f"br_n080_s{seed}" / f"sw_branch_n080_s{seed}" / "all_results.json"
        if own.exists():
            return _load(own)
        return None if seed == 42 else _load(RES / f"ms_stage2_s{seed}" / "all_results.json")
    if n1 == 160:
        return _load(KBS / f"ext_s{seed}" / f"sw_stage1ext_s{seed}" / "all_results.json")
    return _load(KBS / f"br_n{n1:03d}_s{seed}" / f"sw_branch_n{n1:03d}_s{seed}" / "all_results.json")


def ours_run(seed: int) -> list[dict] | None:
    """The paper's two-stage pipeline run (N1 = 80) for this seed."""
    if seed == 42:
        return _load(RES / "geo21_warm_geoaware_s42" / "all_results.json")
    return _load(RES / f"ms_stage2_s{seed}" / "all_results.json")


def baseline_run(method: str, seed: int) -> list[dict] | None:
    for p in (KBS / f"bl_{method}_s{seed}" / "mobo_baselines" / f"{method}_seed{seed}.json",
              RES / "mobo_baselines" / f"{method}_seed{seed}.json"):
        d = _load(p)
        if d is not None:
            return d.get("records", d)
    return None


# ------------------------------------------------------------------------ metrics
def objs_ours(recs: list[dict]) -> np.ndarray:
    return np.array([[r["objectives"][k] for k in OBJ] for r in recs], dtype=float)


def feas_ours(recs: list[dict]) -> np.ndarray:
    return np.array([bool(r.get("feasible")) and not r.get("rejected") for r in recs])


def objs_bl(recs: list[dict]) -> np.ndarray:
    return np.array([[r.get("pae_pct") or 0.0, r.get("gain_db") or 0.0, r.get("psat_dbm") or 0.0]
                     for r in recs], dtype=float)


def feas_bl(recs: list[dict]) -> np.ndarray:
    return np.array([bool(r.get("is_feasible")) and not r.get("rejected") for r in recs])


def hv(Y: np.ndarray, ref: np.ndarray) -> float:
    if len(Y) == 0:
        return 0.0
    part = FastNondominatedPartitioning(ref_point=torch.tensor(ref, dtype=torch.double),
                                        Y=torch.tensor(Y, dtype=torch.double))
    return float(part.compute_hypervolume().item())


def refpt(Y: np.ndarray) -> np.ndarray:
    lo, hi = Y.min(0), Y.max(0)
    return lo - 0.1 * np.maximum(hi - lo, 1e-6)


def stats(Y: np.ndarray, F: np.ndarray, ref: np.ndarray, n: int) -> dict:
    Y, F = Y[:n], F[:n]
    fe = Y[F]
    return {"n": len(Y), "feas": int(F.sum()),
            "best_pae": float(fe[:, 0].max()) if len(fe) else float("nan"),
            "hv": hv(fe, ref)}


# -------------------------------------------------------------------------- sweep
def analyze_sweep() -> dict:
    out: dict = {}
    print("\n=== Stage-I budget sweep (180-evaluation budget; HV ref per seed over union of all conditions)")
    print(f"{'seed':>5} {'N1':>4} {'n':>4} {'feas':>5} {'bestPAE':>8} {'HV':>10}")
    for s in SEEDS:
        runs = {n1: sweep_run(s, n1) for n1 in N1_LIST}
        runs["paper"] = ours_run(s)   # the August two-stage run (N1=80) reported in the paper
        avail = {n1: r for n1, r in runs.items() if r}
        if not avail:
            continue
        Yall = np.vstack([objs_ours(r)[feas_ours(r)] for r in avail.values() if feas_ours(r).any()]
                         or [np.zeros((1, 3))])
        ref = refpt(Yall)
        out[s] = {}
        for n1, r in avail.items():
            st = stats(objs_ours(r), feas_ours(r), ref, BUDGET)
            st["complete"] = len(r) >= BUDGET
            out[s][n1] = st
            flag = "" if st["complete"] else f"  (partial: {len(r)}/{BUDGET})"
            print(f"{s:>5} {str(n1):>5} {st['n']:>4} {st['feas']:>5} {st['best_pae']:>8.1f} {st['hv']:>10.0f}{flag}")
    # medians across seeds per N1
    print("\nmedian over seeds (complete runs only):")
    print(f"{'N1':>4} {'nseeds':>6} {'feas':>5} {'bestPAE':>8} {'HV':>10}")
    for n1 in N1_LIST:
        vals = [out[s][n1] for s in out if n1 in out[s] and out[s][n1]["complete"]]
        if vals:
            print(f"{n1:>4} {len(vals):>6} {np.median([v['feas'] for v in vals]):>5.0f} "
                  f"{np.nanmedian([v['best_pae'] for v in vals]):>8.1f} {np.median([v['hv'] for v in vals]):>10.0f}")
    return out


# ---------------------------------------------------------------------- baselines
def analyze_baselines() -> dict:
    out: dict = {}
    print("\n=== External optimizers vs. the proposed pipeline (per seed; HV ref over union of all methods)")
    print(f"{'seed':>5} {'method':<14} {'n':>4} {'rej':>4} {'feas@100':>8} {'feas@180':>8} {'PAE@100':>8} {'PAE@180':>8} {'HV@180':>10}")
    for s in SEEDS:
        runs: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        o = ours_run(s)
        if o:
            runs["proposed"] = (objs_ours(o), feas_ours(o), sum(1 for r in o if r.get("rejected")))
        for m in BASELINES:
            b = baseline_run(m, s)
            if b:
                runs[m] = (objs_bl(b), feas_bl(b), sum(1 for r in b if r.get("rejected")))
        if not runs:
            continue
        Yall = np.vstack([Y[F] for Y, F, _ in runs.values() if F.any()] or [np.zeros((1, 3))])
        ref = refpt(Yall)
        out[s] = {}
        for m, (Y, F, rej) in runs.items():
            a, b = stats(Y, F, ref, 100), stats(Y, F, ref, BUDGET)
            out[s][m] = {"n": len(Y), "rej": rej, "feas100": a["feas"], "feas180": b["feas"],
                         "pae100": a["best_pae"], "pae180": b["best_pae"], "hv180": b["hv"],
                         "complete": len(Y) >= BUDGET}
            flag = "" if len(Y) >= BUDGET else f"  (partial: {len(Y)}/{BUDGET})"
            print(f"{s:>5} {m:<14} {len(Y):>4} {rej:>4} {a['feas']:>8} {b['feas']:>8} "
                  f"{a['best_pae']:>8.1f} {b['best_pae']:>8.1f} {b['hv']:>10.0f}{flag}")
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if mode in ("sweep", "all"):
        res["sweep"] = analyze_sweep()
    if mode in ("baselines", "all"):
        res["baselines"] = analyze_baselines()
    outp = BASE / "data" / "kbs" / "analysis_summary.json"
    json.dump(res, open(outp, "w"), indent=1, default=lambda x: None if x != x else x)
    print(f"\nwritten {outp}")


if __name__ == "__main__":
    main()

"""Convergence of the compared optimizers (R2-2/R5-5): hypervolume of the
feasible set versus evaluations, normalized per seed by the largest final value
among the methods, median over the five seeds with the inter-seed range.

    python3 plots/generate_convergence.py

Output: plots/paper/fig_convergence.{png,svg}
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "campaign_kbs"))
import analyze_kbs as A  # noqa: E402

OUTDIR = os.path.join(BASE, "plots", "paper")
METHODS = [("proposed", "Proposed pipeline", "#D55E00", "-"),
           ("hebo_general", "HEBO", "#0072B2", "-"),
           ("ax_mobo", "Ax qNEHVI", "#009E73", "-"),
           ("optuna_motpe", "Optuna MOTPE", "#CC79A7", "-"),
           ("pymoo_nsga2", "NSGA-II", "#E69F00", "-"),
           ("pymoo_nsga3", "NSGA-III", "#56B4E9", "--"),
           ("pymoo_moead", "MOEA/D", "#999999", "--"),
           ("pibo", r"$\pi$BO-qLogNEHVI", "#000000", "-.")]
KS = np.arange(10, A.BUDGET + 1, 5)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 1000, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})


def runs_for_seed(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    o = A.ours_run(seed)
    if o:
        out["proposed"] = (A.objs_ours(o), A.feas_ours(o))
    for m in A.BASELINES:
        b = A.baseline_run(m, seed)
        if b and len(b) >= A.BUDGET:
            out[m] = (A.objs_bl(b), A.feas_bl(b))
    p = A.pibo_run(seed)
    if p and len(p) >= A.BUDGET:
        out["pibo"] = (A.objs_ours(p), A.feas_ours(p))
    return out


def traces(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    runs = runs_for_seed(seed)
    Yall = np.vstack([Y[F] for Y, F in runs.values() if F.any()] or [np.zeros((1, 3))])
    ref = A.refpt(Yall)
    best_final = max(A.hv(Y[F], ref) for Y, F in runs.values())
    out = {}
    for m, (Y, F) in runs.items():
        hv = np.array([A.hv(Y[:k][F[:k]], ref) for k in KS]) / (best_final if best_final else np.nan)
        nf = np.array([F[:k].sum() for k in KS])
        out[m] = (hv, nf)
    return out


def main() -> None:
    per_seed = {s: traces(s) for s in A.SEEDS}
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.0))
    for key, label, col, ls in METHODS:
        H = [per_seed[s][key][0] for s in A.SEEDS if key in per_seed[s]]
        N = [per_seed[s][key][1] for s in A.SEEDS if key in per_seed[s]]
        if not H:
            continue
        H, N = np.array(H), np.array(N)
        n = len(H)
        med = np.nanmedian(H, axis=0)
        ax.plot(KS, med, ls, color=col, lw=2, label=f"{label} ({n} seeds)")
        if n >= 3:
            ax.fill_between(KS, np.nanmin(H, axis=0), np.nanmax(H, axis=0), color=col, alpha=0.10, lw=0)
    ax.axhline(1.0, color="0.4", ls=":", lw=0.8)
    ax.set_ylabel("HV / best final HV")
    ax.set_xlabel("Evaluations")
    ax.set_xlim(0, A.BUDGET)
    ax.set_ylim(0, 1.05)
    ax.axvline(100, color="0.6", ls="--", lw=0.7)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"fig_convergence.{ext}"))
    print("saved", os.path.join(OUTDIR, "fig_convergence.png"))


if __name__ == "__main__":
    main()

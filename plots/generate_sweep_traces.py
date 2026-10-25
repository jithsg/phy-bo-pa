"""Per-seed traces of the best feasible PAE versus evaluations for selected
hand-over points of the Stage-I budget sweep (R3-2/R5-2): shows the cold-start
behaviour at N1=0 and that the physics stage, once warm-started, continues
like the zero-mean search.

    python3 plots/generate_sweep_traces.py

Output: plots/paper/fig_sweep_traces.{png,svg}
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
SHOW = [(0, "#CC79A7"), (20, "#009E73"), (80, "#D55E00"), (160, "#0072B2")]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 1000, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})


def best_trace(recs: list[dict]) -> np.ndarray:
    best, out = np.nan, []
    for r in recs[:A.BUDGET]:
        if r.get("feasible") and not r.get("rejected"):
            v = r["objectives"]["PAE_pct"]
            best = v if np.isnan(best) else max(best, v)
        out.append(best)
    return np.array(out)


def main() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.2), sharey=True)
    for ax, s in zip(axes, A.SEEDS):
        for n1, col in SHOW:
            r = A.sweep_run(s, n1)
            if not r:
                continue
            t = best_trace(r)
            ax.plot(np.arange(1, len(t) + 1), t, color=col, lw=1.4,
                    label=f"$N_1$={n1}" + (" (no priors)" if n1 == 160 else ""))
            if 0 < n1 < 160:
                ax.axvline(A.N_DOE + n1, color=col, ls=":", lw=0.8, alpha=0.8)
        # standard BO with no physical information at all: zero-mean GP, no d_in inequality
        nodin = A._load(A.RES / f"cabl_nodin_ext_s{s}" / "all_results.json")
        if nodin:
            t = best_trace(nodin)
            ax.plot(np.arange(1, len(t) + 1), t, color="0.25", lw=1.2, ls="--",
                    label="no physics (no priors, no $d_\\mathrm{in}$)")
        ax.axvline(A.N_DOE, color="0.5", ls="--", lw=0.6)
        ax.set_title(f"seed {s}", fontsize=9)
        ax.set_xlabel("Evaluations")
        ax.set_xlim(0, A.BUDGET)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Best feasible PAE (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"fig_sweep_traces.{ext}"))
    print("saved", os.path.join(OUTDIR, "fig_sweep_traces.png"))


if __name__ == "__main__":
    main()

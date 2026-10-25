"""Stage-I budget sweep figure (R3-2/R3-3/R5-1..3): best feasible PAE and
hypervolume at the 180-evaluation budget versus the number of zero-mean
iterations N1 before the physics-informed stage, one thin line per seed and
the median across seeds.

    python3 plots/generate_sweep_fig.py        (reads campaign_kbs/analysis_summary.json)

Output: plots/paper/fig_sweep_n1.{png,svg}
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(BASE, "data", "kbs", "analysis_summary.json")
OUTDIR = os.path.join(BASE, "plots", "paper")
SEEDS = ["42", "7", "11", "23", "101"]
N1 = [0, 20, 40, 60, 80, 100, 120, 160]
SEED_COL = {"42": "#D55E00", "7": "#0072B2", "11": "#009E73", "23": "#CC79A7", "101": "#E69F00"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 1000, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})


def series(sw: dict, seed: str, key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for n1 in N1:
        st = sw.get(seed, {}).get(str(n1))
        if st and st.get("complete"):
            xs.append(n1)
            ys.append(st[key])
    return xs, ys


def main() -> None:
    sw = json.load(open(SUMMARY))["sweep"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, key, lab in zip(axes, ("best_pae", "hv", "feas"),
                            ("Best feasible PAE (%)", "Hypervolume (per-seed ref.)", "Feasible designs (/180)")):
        med_x, med_y = [], []
        for n1 in N1:
            vals = [sw[s][str(n1)][key] for s in SEEDS
                    if str(n1) in sw.get(s, {}) and sw[s][str(n1)].get("complete")]
            if len(vals) >= 3:
                med_x.append(n1)
                med_y.append(float(np.nanmedian(vals)))
        for s in SEEDS:
            xs, ys = series(sw, s, key)
            if xs:
                ax.plot(xs, ys, "-o", color=SEED_COL[s], lw=0.9, ms=3, alpha=0.75, label=f"seed {s}")
        if med_x:
            ax.plot(med_x, med_y, "-", color="black", lw=2.2, label="median")
        ax.axvline(80, color="0.4", ls=":", lw=0.9)
        ax.set_xlabel("$N_1$")
        ax.set_ylabel(lab, fontsize=9)
        ax.set_xticks(N1)
        ax.tick_params(labelsize=7.5)
    axes[0].text(82, axes[0].get_ylim()[0] + 0.5, "paper\n(80/80)", fontsize=7, color="0.3", va="bottom")
    fig.text(0.5, 0.075, "$N_1$ = zero-mean BO iterations before the physics-informed stage (total budget fixed at 20 DoE + 160 BO)",
             ha="center", fontsize=8, color="0.3")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"fig_sweep_n1.{ext}"))
    print("saved", os.path.join(OUTDIR, "fig_sweep_n1.png"))


if __name__ == "__main__":
    main()

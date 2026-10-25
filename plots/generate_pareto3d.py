"""3-D Pareto fronts (PAE, Gain, Psat) of the feasible designs found by each
compared optimizer at the 180-evaluation budget (R5-7).

    python3 plots/generate_pareto3d.py [seed]      (default 42)

Output: plots/paper/fig_pareto3d_s{seed}.{png,svg}
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
# fixed categorical order/colours (Okabe-Ito), never cycled
METHODS = [("proposed", "Proposed pipeline", "#D55E00", "o"),
           ("hebo_general", "HEBO", "#0072B2", "s"),
           ("ax_mobo", "Ax qNEHVI", "#009E73", "^"),
           ("optuna_motpe", "Optuna MOTPE", "#CC79A7", "D"),
           ("pymoo_nsga2", "NSGA-II", "#E69F00", "v"),
           ("pymoo_nsga3", "NSGA-III", "#56B4E9", "<"),
           ("pymoo_moead", "MOEA/D", "#999999", ">")]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 1000, "savefig.bbox": "tight",
})


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Non-dominated mask for maximization of all columns."""
    n = len(Y)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if keep[i]:
            dom = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
            if dom.any():
                keep[i] = False
    return keep


def load(seed: int) -> dict[str, np.ndarray]:
    data = {}
    o = A.ours_run(seed)
    if o:
        Y, F = A.objs_ours(o), A.feas_ours(o)
        data["proposed"] = Y[:A.BUDGET][F[:A.BUDGET]]
    for m in A.BASELINES:
        b = A.baseline_run(m, seed)
        if b:
            Y, F = A.objs_bl(b), A.feas_bl(b)
            data[m] = Y[:A.BUDGET][F[:A.BUDGET]]
    return data


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    data = load(seed)
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for key, label, col, mk in METHODS:
        Y = data.get(key)
        if Y is None or len(Y) == 0:
            continue
        pm = pareto_mask(Y)
        P = Y[pm]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=34, marker=mk, color=col, alpha=0.95,
                   edgecolors="white", linewidths=0.5,
                   label=f"{label}: {pm.sum()} Pareto / {len(Y)} feasible")
        # drop lines to the floor make depth readable
        z0 = 15.1
        for x, y, z in P:
            ax.plot([x, x], [y, y], [z0, z], color=col, alpha=0.25, lw=0.6)
    ax.set_xlabel("PAE (%)", labelpad=4)
    ax.set_ylabel("Gain (dB)", labelpad=4)
    ax.set_zlabel("$P_\\mathrm{sat}$ (dBm)", labelpad=4)
    ax.set_zlim(15.1, None)
    ax.view_init(elev=20, azim=-55)
    ax.tick_params(labelsize=7.5, pad=1)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"fig_pareto3d_s{seed}.{ext}"))
    print("saved", os.path.join(OUTDIR, f"fig_pareto3d_s{seed}.png"))


if __name__ == "__main__":
    main()

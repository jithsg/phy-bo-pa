"""Train the cheap-fidelity S-parameter prior (differentiable MLP).

Data: results/sp_batch.csv (SP-only simulations at snapped geometry).
Model: 7 -> 64 -> 64 -> 4 MLP (double precision) predicting
       [S11_dB, S22_dB, log10(K), B1], inputs normalized by BOUNDS.
Eval:  (i) 80/20 holdout R^2 on the cheap-sim data;
       (ii) cross-fidelity R^2 against every cached FULL simulation
            (~1,450 designs across all campaigns), inputs snapped.

Saves weights + normalization to results/sparam_mlp.pt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from bo_loop_geo import snap_to_pdk_grid  # noqa: E402
from config import BOUNDS_LOWER, BOUNDS_UPPER  # noqa: E402

RESULTS = BASE / "results"
CSV = RESULTS / "sp_batch.csv"
OUT = RESULTS / "sparam_mlp.pt"
XCOLS = ["L1L2_rad", "L1L2_nr", "L3L4_rad", "L3L4_nr", "c_wr", "r_l", "m_wr"]
TARGETS = ["S11_dB", "S22_dB", "logK", "B1"]
CLIP_DB = -25.0
FULL_RUNS = [
    ("geo21_warm_geoaware_s42", None), ("ablA_warmonly_s42", None),
    ("ablB_physscratch_s42", None),
    *[(f"ms_stage1_s{s}", None) for s in (7, 11, 23, 101)],
    *[(f"ms_stage2_s{s}", None) for s in (7, 11, 23, 101)],
]


def make_mlp() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(7, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 4),
    ).double()


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - BOUNDS_LOWER) / (BOUNDS_UPPER - BOUNDS_LOWER)


def targets_from(s11, s22, k, b1) -> np.ndarray:
    return np.column_stack([
        np.clip(s11, CLIP_DB, None), np.clip(s22, CLIP_DB, None),
        np.log10(np.clip(k, 1e-3, 1e8)), np.clip(b1, -2.0, 3.0),
    ])


def r2(pred: np.ndarray, true: np.ndarray) -> list[float]:
    out = []
    for j in range(pred.shape[1]):
        ss_res = ((pred[:, j] - true[:, j]) ** 2).sum()
        ss_tot = ((true[:, j] - true[:, j].mean()) ** 2).sum()
        out.append(float(1 - ss_res / ss_tot))
    return out


def load_full_corpus() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for run, _ in FULL_RUNS:
        for e in json.load(open(RESULTS / run / "all_results.json")):
            c = e.get("constraints") or {}
            v = [c.get("S11_dB"), c.get("S22_dB"), c.get("K"), c.get("B1")]
            if any(u is None for u in v) or not (-60 < v[0] < 0):
                continue
            xs.append(snap_to_pdk_grid(np.array(e["x"], dtype=float)))
            ys.append(v)
    ys = np.array(ys)
    return np.array(xs), targets_from(ys[:, 0], ys[:, 1], ys[:, 2], ys[:, 3])


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["error"].isna() | (df["error"] == "")]
    df = df.dropna(subset=["S11_dB", "S22_dB", "K", "B1"])
    print(f"cheap-sim training rows: {len(df)}")

    X = normalize(df[XCOLS].to_numpy(float))
    Y = targets_from(df["S11_dB"], df["S22_dB"], df["K"], df["B1"])
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-9

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]

    def fit(Xtr, Ytr, steps=4000):
        net = make_mlp()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
        xt = torch.tensor(Xtr)
        yt = torch.tensor((Ytr - y_mu) / y_sd)
        for i in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.huber_loss(net(xt), yt, delta=1.0)
            loss.backward()
            opt.step()
        return net

    net = fit(X[tr], Y[tr])
    with torch.no_grad():
        pred_te = net(torch.tensor(X[te])).numpy() * y_sd + y_mu
    print("holdout R2 (cheap-sim):",
          dict(zip(TARGETS, [round(v, 3) for v in r2(pred_te, Y[te])])))

    Xf, Yf = load_full_corpus()
    with torch.no_grad():
        pred_f = net(torch.tensor(normalize(Xf))).numpy() * y_sd + y_mu
    print(f"cross-fidelity R2 vs {len(Xf)} full sims:",
          dict(zip(TARGETS, [round(v, 3) for v in r2(pred_f, Yf)])))

    net_full = fit(X, Y)
    torch.save({"state_dict": net_full.state_dict(),
                "y_mu": y_mu, "y_sd": y_sd, "targets": TARGETS,
                "clip_db": CLIP_DB}, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

"""Calibrate and validate the physics-structured S11/S22 priors offline.

Protocol (no simulations):
- TRAIN: the 100 evaluations available at Stage-II entry of the primary
  campaign (geo21[:100]) — exactly the data the optimizer would have had.
- TEST: every other cached evaluation across all runs (~1,300 designs the
  calibration never saw).
- Metric: R^2 on dB values clipped at -25 dB (deep notches are chaotic),
  compared against the incumbent linear-regression prior fitted on the same
  training split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from physics_sparam import calibrate, s11_db_model, s22_db_model, save_theta

RESULTS = Path(__file__).parent / "results"
TRAIN_RUN = "geo21_warm_geoaware_s42"
TEST_RUNS = [
    "ablA_warmonly_s42", "ablB_physscratch_s42",
    "ms_stage1_s7", "ms_stage1_s11", "ms_stage1_s23", "ms_stage1_s101",
    "ms_stage2_s7", "ms_stage2_s11", "ms_stage2_s23", "ms_stage2_s101",
]
CLIP = -25.0


def load(run: str, lo: int = 0, hi: int | None = None):
    rows = json.load(open(RESULTS / run / "all_results.json"))[lo:hi]
    x, s11, s22 = [], [], []
    for e in rows:
        c = e.get("constraints") or {}
        v11, v22 = c.get("S11_dB"), c.get("S22_dB")
        if v11 is None or v22 is None or not (-60 < v11 < 0) or not (-60 < v22 < 0):
            continue
        x.append(e["x"]); s11.append(v11); s22.append(v22)
    return (torch.tensor(np.array(x), dtype=torch.double),
            torch.tensor(s11, dtype=torch.double),
            torch.tensor(s22, dtype=torch.double))


def r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    p = torch.clamp(pred, min=CLIP).numpy()
    t = torch.clamp(true, min=CLIP).numpy()
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def linreg_baseline(xtr, ytr, xte):
    A = np.hstack([xtr.numpy(), np.ones((len(xtr), 1))])
    w, *_ = np.linalg.lstsq(A, np.clip(ytr.numpy(), CLIP, None), rcond=None)
    Ate = np.hstack([xte.numpy(), np.ones((len(xte), 1))])
    return torch.tensor(Ate @ w)


def main() -> None:
    xtr, s11tr, s22tr = load(TRAIN_RUN, 0, 100)
    xte_list = [load(r) for r in TEST_RUNS]
    # exclude geo21's own stage II from test? include it: unseen by training
    xte2, s11te2, s22te2 = load(TRAIN_RUN, 100, None)
    xte = torch.cat([xte2] + [t[0] for t in xte_list])
    s11te = torch.cat([s11te2] + [t[1] for t in xte_list])
    s22te = torch.cat([s22te2] + [t[2] for t in xte_list])
    print(f"train n={len(xtr)}  test n={len(xte)}")

    theta = calibrate(xtr, s11tr, s22tr)
    save_theta(theta)
    print("calibrated theta:", {k: round(v, 4) for k, v in theta.items()})

    th = {k: torch.tensor(v, dtype=torch.double) for k, v in theta.items()}
    print("\n            train R2   test R2")
    for name, model, ytr, yte in (
        ("S11 physics", s11_db_model, s11tr, s11te),
        ("S22 physics", s22_db_model, s22tr, s22te),
    ):
        print(f"{name:>12}: {r2(model(xtr, th), ytr):8.3f} {r2(model(xte, th), yte):9.3f}")
    for name, ytr, yte in (("S11 linreg", s11tr, s11te), ("S22 linreg", s22tr, s22te)):
        pr_tr = linreg_baseline(xtr, ytr, xtr)
        pr_te = linreg_baseline(xtr, ytr, xte)
        print(f"{name:>12}: {r2(pr_tr, ytr):8.3f} {r2(pr_te, yte):9.3f}")


if __name__ == "__main__":
    main()

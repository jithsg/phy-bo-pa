"""Round-2 cheap-fidelity batch: densify the matched region.

Round 1 (uniform Sobol) covered the badly-matched bulk of the space but put
only 22/1000 points below S11 = -8 dB, leaving the surrogate blind exactly
where BO operates. This round targets that region:

- 300 designs: Gaussian perturbations (sigma = 5% of each range) around the
  unique feasible designs from all full-sim campaigns;
- 100 designs: perturbations (sigma = 3%) around random Stage-II BO points
  (the region acquisition actually visits).

All snapped, d_in-valid, deduplicated against round 1 and each other.
Appends to the same results/sp_batch.csv with indices starting at 1000.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from bo_loop_geo import snap_to_pdk_grid
from config import BOUNDS_LOWER, BOUNDS_UPPER, check_d_in, IND_W, IND_SPACING
from run_sp_batch import CHUNK, COLS, OUT_CSV, run_chunk

RESULTS = BASE / "results"
RUNS = ["geo21_warm_geoaware_s42", "ablA_warmonly_s42", "ablB_physscratch_s42",
        *[f"ms_stage1_s{s}" for s in (7, 11, 23, 101)],
        *[f"ms_stage2_s{s}" for s in (7, 11, 23, 101)]]
N_FEAS_PERT, N_BO_PERT = 300, 100
IDX0 = 1000


def corpus_points() -> tuple[np.ndarray, np.ndarray]:
    feas, stage2 = [], []
    for run in RUNS:
        rows = json.load(open(RESULTS / run / "all_results.json"))
        for i, e in enumerate(rows):
            x = np.array(e["x"], dtype=float)
            if e.get("feasible"):
                feas.append(x)
            if i >= 100 and "stage2" in run or run.startswith("geo21") and i >= 100:
                stage2.append(x)
    return np.array(feas), np.array(stage2)


def perturb(centers: np.ndarray, n: int, sigma_frac: float,
            rng: np.random.Generator, seen: set) -> list[np.ndarray]:
    span = BOUNDS_UPPER - BOUNDS_LOWER
    out = []
    while len(out) < n:
        c = centers[rng.integers(0, len(centers))]
        x = np.clip(c + rng.normal(0, sigma_frac * span),
                    BOUNDS_LOWER, BOUNDS_UPPER)
        x = snap_to_pdk_grid(x)
        key = tuple(np.round(x, 9))
        if key in seen:
            continue
        if (check_d_in(x[0], x[1], IND_W, IND_SPACING)
                and check_d_in(x[2], x[3], IND_W, IND_SPACING)):
            seen.add(key)
            out.append(x)
    return out


def main() -> None:
    prev = pd.read_csv(OUT_CSV)
    seen = {tuple(np.round(r, 9)) for r in prev[COLS[1:8]].to_numpy(float)}
    rng = np.random.default_rng(7)
    feas, stage2 = corpus_points()
    print(f"centers: {len(feas)} feasible, {len(stage2)} stage-II points")
    designs = (perturb(feas, N_FEAS_PERT, 0.05, rng, seen)
               + perturb(stage2, N_BO_PERT, 0.03, rng, seen))
    done = set(prev["idx"].astype(int))
    todo = [(IDX0 + i, d) for i, d in enumerate(designs)
            if (IDX0 + i) not in done]
    print(f"round-2 designs to run: {len(todo)}")
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        rows = run_chunk(chunk)
        pd.DataFrame(rows)[COLS].to_csv(OUT_CSV, mode="a", header=False,
                                        index=False)
        n_ok = sum(1 for r in rows if not r["error"])
        print(f"[chunk] {chunk[0][0]:4d}-{chunk[-1][0]:4d}: {n_ok}/{len(rows)} ok",
              flush=True)
    print("SP BATCH-2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()

"""Replay the optimizer step of a finished run on THIS machine for timing only.

For BO iteration n the data available before proposal n are the first 20+(n-1)
records; the replay fits the nine zero-mean GPs and optimizes the acquisition with
the run's own settings (q=1, restarts 5, raw 256, MC 128, d_in inequality) and
records the wall time of that step.  No Spectre call is made and nothing is
evaluated, so the replayed proposals are not used for anything but timing.

    replay_timing.py pibo <seed> [step]      πBO-style run   (results/pibo_s{seed})
    replay_timing.py control <seed> [step]   zero-mean control (kbs_results/ext_s{seed})

step (default 4) replays n = 1, 1+step, ... up to 160.  Output:
results/pibo/replay_{kind}_s{seed}.json = {"host":..., "times": {n: seconds}}.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from botorch.utils.transforms import normalize

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import bo_loop_geo as G  # noqa: E402
from config import LOCAL_RESULTS_DIR, N_DIMS, OptimizationConfig  # noqa: E402
from pibo_prior import PriorSpec  # noqa: E402

RES = Path(LOCAL_RESULTS_DIR)
N_DOE, N_IT = 20, 160


def load(kind: str, seed: int) -> list[dict]:
    if kind == "pibo":
        p = RES / f"pibo_s{seed}" / "all_results.json"
    else:
        p = RES / "kbs_results" / f"ext_s{seed}" / f"sw_stage1ext_s{seed}" / "all_results.json"
    return json.load(open(p))


def ref_from(recs: list[dict]) -> torch.Tensor:
    Y = np.array([[r["objectives"][k] for k in G.OBJ_NAMES] for r in recs])
    ref = [Y[:, j].min() - 0.1 * max(Y[:, j].max() - Y[:, j].min(), 1e-6) for j in range(3)]
    return torch.tensor(ref, dtype=G.DTYPE)


def step_time(recs: list[dict], ref: torch.Tensor, cfg: OptimizationConfig, prior, n: int) -> float:
    valid = [r for r in recs if not r.get("rejected")]
    X = np.array([r["x"] for r in valid])
    Yo = np.array([[r["objectives"][k] for k in G.OBJ_NAMES] for r in valid])
    Yc_raw = np.array([[r["constraints"][nm] for nm, _, _ in G.CONSTRAINT_TRANSFORMS] for r in valid])
    Yc = np.empty_like(Yc_raw)
    for ci, (_, sc, off) in enumerate(G.CONSTRAINT_TRANSFORMS):
        Yc[:, ci] = Yc_raw[:, ci] * sc + off
    bounds_norm = torch.tensor([[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=G.DTYPE)
    t0 = time.time()
    train_X = normalize(torch.tensor(X, dtype=G.DTYPE), G.BOUNDS)
    train_Y = torch.tensor(np.hstack([Yo, Yc]), dtype=G.DTYPE)
    model, _ = G.build_model_list(train_X, train_Y, use_physics=False, use_physics_features=False)
    G.get_candidates(model, train_X, ref, bounds_norm, cfg, use_din_constraint=True,
                     pibo_prior=prior, pibo_n=n)
    return time.time() - t0


def main() -> None:
    kind, seed = sys.argv[1], int(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    recs = load(kind, seed)
    if len(recs) < N_DOE + N_IT:
        raise SystemExit(f"{kind} seed {seed}: only {len(recs)} records")
    prior = PriorSpec.manual_design(N_IT) if kind == "pibo" else None
    cfg = OptimizationConfig(n_initial=N_DOE, n_iterations=N_IT, batch_size=1, seed=seed,
                             num_restarts=5, raw_samples=256)
    ref_doe = ref_from(recs[:N_DOE])
    ref_100 = ref_from(recs[:100])          # the control was resumed at 100 points
    out = RES / "pibo" / f"replay_{kind}_s{seed}.json"
    out.parent.mkdir(exist_ok=True)
    times = json.load(open(out))["times"] if out.exists() else {}
    torch.manual_seed(seed)
    for n in range(1, N_IT + 1, step):
        if str(n) in times:
            continue
        ref = ref_doe if (kind == "pibo" or n <= 80) else ref_100
        t = step_time(recs[:N_DOE + n - 1], ref, cfg, prior, n)
        times[str(n)] = t
        print(f"{kind} s{seed} n={n:3d} data={N_DOE + n - 1:3d} t_opt={t:6.1f} s", flush=True)
        json.dump({"host": platform.node(), "kind": kind, "seed": seed, "step": step, "times": times},
                  open(out, "w"), indent=1)
    print(f"written {out}: {len(times)} iterations, median {np.median(list(times.values())):.1f} s")


if __name__ == "__main__":
    main()

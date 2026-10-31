"""πBO-style prior-weighted qLogNEHVI baseline lane (reviewer request, KBS revision).

Usage:
    pibo_lane.py <seed>            seeds 42, 7, 11, 23, 101

Matched to the zero-mean reference (results/kbs_results/ext_s{seed}, Table 18
N1 = 160 row): same 20-point DoE (copied from that run, so its 20 simulations are
reused), zero-mean GPs, d_in inequality, PDK snapping, q = 1, num_restarts = 5,
raw_samples = 256, mc_samples = 128, 160 BO iterations (180 evaluations in total),
use_physics = False.  The only change is the acquisition: log qLogNEHVI(x) +
(β/n) log π(x) with β = 16 (= N/10) and π a truncated Gaussian centred on the
manual design (config.PRESIM_X), σ = 25 % of each domain, floor ε = 1e-12.
Point ids: pibo_s{seed}_batch{iter:03d}_pt0.  Results: results/pibo_s{seed}/.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from config import LOCAL_RESULTS_DIR  # noqa: E402
from bo_loop_geo import run_bo  # noqa: E402
from pibo_prior import PriorSpec  # noqa: E402

RESULTS = Path(LOCAL_RESULTS_DIR)
N_DOE = 20
N_TOTAL_ITERS = 160
SEEDS = (42, 7, 11, 23, 101)


def control_results(seed: int) -> Path:
    """Zero-mean reference run (180 points) whose first 20 points are the DoE."""
    return RESULTS / "kbs_results" / f"ext_s{seed}" / f"sw_stage1ext_s{seed}" / "all_results.json"


def lane_dir(seed: int) -> Path:
    return RESULTS / f"pibo_s{seed}"


def doe_prefix(seed: int) -> Path:
    out = lane_dir(seed)
    prefix = out / "resume_prefix.json"
    if prefix.exists():          # pre-generated on the master and shipped to a worker
        return prefix
    src = control_results(seed)
    if not src.exists():
        raise FileNotFoundError(f"zero-mean reference missing for seed {seed}: {src}")
    with open(src) as f:
        pts = json.load(f)
    if len(pts) < N_DOE:
        raise RuntimeError(f"{src} has {len(pts)} points, need {N_DOE}")
    out = lane_dir(seed)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "resume_prefix.json"
    with open(prefix, "w") as f:
        json.dump(pts[:N_DOE], f)
    return prefix


def lane(seed: int) -> str:
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}, got {seed}")
    out = lane_dir(seed)
    own = out / "all_results.json"
    resume = own if own.exists() else doe_prefix(seed)
    prior = PriorSpec.manual_design(n_iterations=N_TOTAL_ITERS)
    t0 = time.time()
    print(f"\n{'=' * 70}\n  ARM piBO seed {seed}  start {time.strftime('%H:%M:%S')}\n"
          f"  prior: {prior.as_dict()}\n{'=' * 70}", flush=True)
    try:
        run_bo(q=1, use_warping=False, use_dkl=False,
               num_restarts=5, raw_samples=256,
               n_initial=N_DOE, n_iterations=N_TOTAL_ITERS, seed=seed,
               log_dir=str(out), use_physics=False, warmstart_centers=None,
               resume_from=str(resume), pid_tag="pibo", pibo_prior=prior)
        status = "OK"
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    print(f"\n  ARM piBO seed {seed} {status} after {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return status


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    seed = int(sys.argv[1])
    status = lane(seed)
    print(f"\nLANE pibo_s{seed} {status}", flush=True)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()

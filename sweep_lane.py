"""Stage-I budget sweep lanes for the KBS revision (R3-2, R3-3, R5-1..R5-3).

Usage:
    sweep_lane.py ext  <seed>          extend the zero-mean Stage I run to 160 BO iters
    sweep_lane.py br   <seed> <N1>     Stage II (physics + d_in) branched after N1
                                       zero-mean iterations, run to 160 total

Per seed the zero-mean Stage I source is the geometry-aware run used in the
paper (ms_stage1_s{seed}; cabl_din_s42 for seed 42, whose original Stage I
predates the geometry-aware loop).  A branch at N1 resumes from that run's
first 20 + N1 points, so every branch shares an identical zero-mean prefix
by construction.  N1 = 0 is the cold-start physics GP (resume from the DoE
only); N1 >= 100 needs the extension lane to have finished first.

Settings match the paper lanes (q=1, num_restarts=5, raw_samples=256).
Point ids: sw1_s{seed}_* (extension) and sw2n{N1}_s{seed}_* (branches).
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

RESULTS = Path(LOCAL_RESULTS_DIR)
N_DOE = 20
N_TOTAL_ITERS = 160
BRANCH_POINTS = (0, 20, 40, 60, 100, 120)


def stage1_source(seed: int) -> Path:
    """Zero-mean, geometry-aware Stage I results for this seed (100 points)."""
    name = "cabl_din_s42" if seed == 42 else f"ms_stage1_s{seed}"
    return RESULTS / name / "all_results.json"


def ext_dir(seed: int) -> Path:
    return RESULTS / f"sw_stage1ext_s{seed}"


def branch_dir(seed: int, n1: int) -> Path:
    return RESULTS / f"sw_branch_n{n1:03d}_s{seed}"


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _prefix_file(seed: int, n1: int) -> Path:
    """Write the first 20 + N1 points of the zero-mean run to a resume file."""
    src = stage1_source(seed) if n1 <= 80 else ext_dir(seed) / "all_results.json"
    if not src.exists():
        raise FileNotFoundError(f"Stage I source missing for seed {seed}, N1={n1}: {src}")
    pts = _load(src)
    need = N_DOE + n1
    if len(pts) < need:
        raise RuntimeError(f"{src} has {len(pts)} points, need {need} for N1={n1}")
    out = branch_dir(seed, n1)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "resume_prefix.json"
    with open(prefix, "w") as f:
        json.dump(pts[:need], f)
    return prefix


def _arm(name: str, **kw) -> str:
    t0 = time.time()
    print(f"\n{'=' * 70}\n  ARM {name}  start {time.strftime('%H:%M:%S')}\n{'=' * 70}",
          flush=True)
    try:
        run_bo(q=1, use_warping=False, use_dkl=False,
               num_restarts=5, raw_samples=256, **kw)
        status = "OK"
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    print(f"\n  ARM {name} {status} after {(time.time() - t0) / 60:.1f} min", flush=True)
    return status


def lane_ext(seed: int) -> str:
    """Zero-mean Stage I continued from 80 to 160 BO iterations."""
    out = ext_dir(seed)
    own = out / "all_results.json"
    resume = own if own.exists() else stage1_source(seed)
    if not resume.exists():
        raise FileNotFoundError(resume)
    return _arm(f"ext seed {seed}",
                n_initial=N_DOE, n_iterations=N_TOTAL_ITERS, seed=seed,
                log_dir=str(out), use_physics=False, warmstart_centers=None,
                resume_from=str(resume), pid_tag="sw1")


def lane_branch(seed: int, n1: int) -> str:
    """Physics + d_in Stage II resumed after N1 zero-mean iterations."""
    if n1 not in BRANCH_POINTS and n1 != 80:
        raise SystemExit(f"N1 must be one of {BRANCH_POINTS} (or 80), got {n1}")
    out = branch_dir(seed, n1)
    own = out / "all_results.json"
    resume = own if own.exists() else _prefix_file(seed, n1)
    return _arm(f"branch N1={n1} seed {seed}",
                n_initial=N_DOE, n_iterations=N_TOTAL_ITERS, seed=seed,
                log_dir=str(out), use_physics=True, warmstart_centers=None,
                resume_from=str(resume), pid_tag=f"sw2n{n1}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    kind, seed = sys.argv[1], int(sys.argv[2])
    if kind == "ext":
        status = lane_ext(seed)
        tag = f"ext_s{seed}"
    elif kind == "br":
        n1 = int(sys.argv[3])
        status = lane_branch(seed, n1)
        tag = f"br_n{n1}_s{seed}"
    else:
        raise SystemExit(f"unknown lane kind {kind}")
    print(f"\nLANE {tag} {status}", flush=True)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()

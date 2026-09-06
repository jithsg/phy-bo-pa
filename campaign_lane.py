"""One lane of the parallelized ablation/multi-seed campaign.

Usage: campaign_lane.py {AB | s7 | s11 | s23 | s101}

Lane AB : resume Arm A (warm-only control) from its own checkpoint, then run
          Arm B (physics-from-scratch).
Lane sN : Stage I (zero-mean, warm-centered DoE) then Stage II (physics,
          resumed from Stage I) for seed N.

All lanes share the /tmp/pa_workstation_sim.lock via simulator.py, so the
workstation never runs more than one Spectre job at a time. Settings are
identical to the paper (q=1, num_restarts=5, raw_samples=256).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from config import PRESIM_X  # noqa: E402
from bo_loop_geo import run_bo  # noqa: E402

RESULTS = BASE / "results"
GEO2 = RESULTS / "geo2_warm_s42" / "all_results.json"


def arm(name: str, **kw) -> None:
    t0 = time.time()
    print(f"\n{'='*70}\n  ARM {name}  start {time.strftime('%H:%M:%S')}\n{'='*70}",
          flush=True)
    try:
        run_bo(q=1, use_warping=False, use_dkl=False,
               num_restarts=5, raw_samples=256, **kw)
        status = "OK"
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    print(f"\n  ARM {name} {status} after {(time.time()-t0)/60:.1f} min",
          flush=True)


def lane_ab() -> None:
    abl_a = RESULTS / "ablA_warmonly_s42" / "all_results.json"
    resume_a = str(abl_a) if abl_a.exists() else str(GEO2)
    arm("A ablA_warmonly (resume)",
        n_initial=20, n_iterations=160, seed=42,
        log_dir=str(RESULTS / "ablA_warmonly_s42"),
        use_physics=False, warmstart_centers=None,
        resume_from=resume_a, pid_tag="ablA_warmonly")
    arm("B ablB_physscratch",
        n_initial=20, n_iterations=80, seed=42,
        log_dir=str(RESULTS / "ablB_physscratch_s42"),
        use_physics=True, warmstart_centers=[PRESIM_X],
        resume_from=None, pid_tag="ablB_physscratch")


def lane_seed(s: int) -> None:
    stage1_dir = RESULTS / f"ms_stage1_s{s}"
    s1_results = stage1_dir / "all_results.json"
    if not (s1_results.exists()
            and len(json.load(open(s1_results))) >= 100):
        arm(f"C stage1 seed {s}",
            n_initial=20, n_iterations=80, seed=s,
            log_dir=str(stage1_dir),
            use_physics=False, warmstart_centers=[PRESIM_X],
            resume_from=str(s1_results) if s1_results.exists() else None,
            pid_tag="ms1")
    if not s1_results.exists():
        print(f"  [skip] stage2 seed {s}: stage1 results missing", flush=True)
        return
    arm(f"C stage2 seed {s}",
        n_initial=20, n_iterations=160, seed=s,
        log_dir=str(RESULTS / f"ms_stage2_s{s}"),
        use_physics=True, warmstart_centers=None,
        resume_from=str(s1_results), pid_tag="ms2")


def main() -> None:
    lane = sys.argv[1]
    if lane == "AB":
        lane_ab()
    elif lane.startswith("s"):
        lane_seed(int(lane[1:]))
    else:
        raise SystemExit(f"unknown lane {lane}")
    print(f"\nLANE {lane} COMPLETE", flush=True)


if __name__ == "__main__":
    main()

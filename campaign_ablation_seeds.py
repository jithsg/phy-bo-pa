"""Overnight campaign: 2x2 ablation completion + multi-seed statistics.

Arms (sequential; each writes its own results/<tag>_s<seed>/ with the patched
config logging that now records use_physics/warmstart/resume_from):

  A. ablA_warmonly  : warm-start from Stage-I data (geo2, 100 pts), physics
                      OFF, 80 BO iters  -> isolates the warm-start effect.
  B. ablB_physscratch: fresh 20-DoE + 80 BO with physics ON, no warm data
                      -> isolates the physics-prior effect.
  C. ms_stage1_s{S} / ms_stage2_s{S} for S in {7, 11, 23, 101}: full
                      two-stage pipeline replicas for mean/std statistics.

Every arm uses the paper settings: q=1, num_restarts=5, raw_samples=256,
Sobol DoE with baseline-centered perturbation half (warmstart_centers).
Failures in one arm do not stop the campaign.
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
SEEDS = (7, 11, 23, 101)


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


def main() -> None:
    # A: warm-only control (100 warm pts, no physics, 80 more iters)
    arm("A ablA_warmonly",
        n_initial=20, n_iterations=160, seed=42,
        log_dir=str(RESULTS / "ablA_warmonly_s42"),
        use_physics=False, warmstart_centers=None,
        resume_from=str(GEO2), pid_tag="ablA_warmonly")

    # B: physics-from-scratch (20 DoE + 80 iters, physics, no warm data)
    arm("B ablB_physscratch",
        n_initial=20, n_iterations=80, seed=42,
        log_dir=str(RESULTS / "ablB_physscratch_s42"),
        use_physics=True, warmstart_centers=[PRESIM_X],
        resume_from=None, pid_tag="ablB_physscratch")

    # C: multi-seed two-stage replicas
    for s in SEEDS:
        stage1_dir = RESULTS / f"ms_stage1_s{s}"
        arm(f"C stage1 seed {s}",
            n_initial=20, n_iterations=80, seed=s,
            log_dir=str(stage1_dir),
            use_physics=False, warmstart_centers=[PRESIM_X],
            resume_from=None, pid_tag="ms1")
        s1_results = stage1_dir / "all_results.json"
        if not s1_results.exists():
            print(f"  [skip] stage2 seed {s}: stage1 results missing", flush=True)
            continue
        n1 = len(json.load(open(s1_results)))
        if n1 < 100:
            print(f"  [warn] stage1 seed {s} has only {n1} evals; "
                  f"continuing stage2 from what exists", flush=True)
        arm(f"C stage2 seed {s}",
            n_initial=20, n_iterations=160, seed=s,
            log_dir=str(RESULTS / f"ms_stage2_s{s}"),
            use_physics=True, warmstart_centers=None,
            resume_from=str(s1_results), pid_tag="ms2")

    print("\nCAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()

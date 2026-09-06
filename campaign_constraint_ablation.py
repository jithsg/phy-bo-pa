"""Matched constraint-only ablation (REMAINING_WORK item 1).

Arm A  (no-d_in): PDK-snapped, zero-mean, cold-start BO with the analytical
                  inner-diameter inequality REMOVED from the acquisition
                  optimizer (invalid proposals are caught post hoc, as in the
                  non-geometry-aware Stage I).
Arm B  (d_in)   : identical run WITH the inequality.  For seeds 7/11/23/101
                  this is exactly results/ms_stage1_s{seed} (same code path,
                  same seed, bit-identical geometry-filtered DoE -- verified by
                  the preflight check below), so only seed 42 needs a fresh
                  Arm-B run.

Everything else is held identical to campaign_lane.py: 20-point geometry-
filtered DoE (10 Sobol + 10 perturbed around the baseline), 80 BO iterations,
q=1, num_restarts=5, raw_samples=256, zero-mean GPs, PDK snapping before
simulation and GP storage, 24.5 mW P_dc guard band.

Usage:
    python3 campaign_constraint_ablation.py            # full campaign, serial
    python3 campaign_constraint_ablation.py --lane 7A 42B   # only these runs
        (tokens <seed><A|B>; run several lanes concurrently in separate
        processes -- simulator.py's cross-lane lock keeps Spectre serialized on
        the workstation while lanes overlap their local GP fitting)
    python3 campaign_constraint_ablation.py --smoke    # 2 DoE + 1 BO iter
    python3 campaign_constraint_ablation.py --check    # DoE identity only
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from config import PRESIM_X  # noqa: E402
from bo_loop_geo import run_bo, generate_initial_doe  # noqa: E402

RESULTS = BASE / "results"

# Run order: the four seeds whose Arm B already exists come first, so paired
# results become available early; seed 42 (both arms) last.
ARM_A_SEEDS = [7, 11, 23, 101, 42]
ARM_B_FRESH_SEEDS = [42]
N_INITIAL, N_ITER = 20, 80


def _preflight_doe_check() -> bool:
    """Confirm the geometry-filtered DoE is bit-identical to the stored
    Arm-B (ms_stage1) DoEs, i.e. both arms start from the same 20 points."""
    ok = True
    for s in (7, 11, 23, 101):
        ref = RESULTS / f"ms_stage1_s{s}" / "doe_results.json"
        if not ref.exists():
            print(f"  [check] seed {s}: {ref} missing")
            ok = False
            continue
        X_ref = np.array([r["x"] for r in json.load(open(ref))])
        X_new = generate_initial_doe(N_INITIAL, s, warmstart_frac=0.5,
                                     warmstart_centers=[PRESIM_X])
        same = X_new.shape == X_ref.shape and np.allclose(X_new, X_ref, atol=1e-9)
        print(f"  [check] seed {s}: DoE identical to ms_stage1_s{s}: {same}")
        ok = ok and same
    return ok


def _arm(name: str, log_dir: Path, **kw) -> str:
    t0 = time.time()
    print(f"\n{'=' * 70}\n  ARM {name}  start {time.strftime('%Y-%m-%d %H:%M:%S')}"
          f"\n{'=' * 70}", flush=True)
    resume = log_dir / "all_results.json"
    if resume.exists() and len(json.load(open(resume))) >= N_INITIAL + N_ITER:
        print(f"  ARM {name} already complete -- skipping", flush=True)
        return "SKIP"
    try:
        run_bo(q=1, use_warping=False, use_dkl=False,
               num_restarts=5, raw_samples=256,
               log_dir=str(log_dir),
               resume_from=str(resume) if resume.exists() else None,
               **kw)
        status = "OK"
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    print(f"\n  ARM {name} {status} after {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return status


def main(argv: list[str]) -> None:
    smoke = "--smoke" in argv
    n_init, n_iter = (2, 1) if smoke else (N_INITIAL, N_ITER)

    print("Preflight: DoE identity check against existing Arm-B runs")
    ok = _preflight_doe_check()
    if "--check" in argv:
        sys.exit(0 if ok else 1)
    if not ok and not smoke:
        print("Preflight failed -- Arm B (ms_stage1) DoEs do not match. Abort.")
        sys.exit(1)

    common = dict(n_initial=n_init, n_iterations=n_iter,
                  use_physics=False, warmstart_centers=[PRESIM_X])
    if smoke:
        _arm("A smoke (no d_in)", RESULTS / "smoke_cabl_nodin_s7", seed=7,
             use_din_constraint=False, pid_tag="smk_cA", **common)
        return

    # Work list: (seed, arm) in default serial order, or the explicit lane.
    work = [(s, "A") for s in ARM_A_SEEDS] + [(s, "B") for s in ARM_B_FRESH_SEEDS]
    if "--lane" in argv:
        toks = " ".join(argv[argv.index("--lane") + 1:]).replace(",", " ").split()
        work = [(int(t[:-1]), t[-1].upper()) for t in toks if t[-1].upper() in "AB"]
        if not work:
            print("--lane needs tokens like 7A 42B")
            sys.exit(2)

    for s, arm in work:
        if arm == "A":
            _arm(f"A no-d_in seed {s}", RESULTS / f"cabl_nodin_s{s}", seed=s,
                 use_din_constraint=False, pid_tag="cA_nodin", **common)
        else:
            _arm(f"B d_in seed {s}", RESULTS / f"cabl_din_s{s}", seed=s,
                 use_din_constraint=True, pid_tag="cB_din", **common)

    tag = "LANE COMPLETE" if "--lane" in argv else "CAMPAIGN COMPLETE"
    print(f"\n{tag} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Pre-registered test: physics-as-REPRESENTATION, one seed per lane.

Usage: campaign_physfeat_cold.py {42|7|11|23|101}

Cold-start BO (20 DoE + 80 iterations) with zero-mean GPs (no belief priors)
but physics-derived kernel features appended to the inputs (reactances and
matching-mismatch metrics; ARD can mute useless features, unlike a mean
function). Controls: the existing cold zero-mean runs without features.
Success criteria identical to the primed-cold test: pooled BO-phase feasible
rate (one-sided Fisher), HV at the 100-eval budget under frozen per-seed
references shared with the control.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from config import PRESIM_X  # noqa: E402
from bo_loop_geo import run_bo  # noqa: E402

RESULTS = BASE / "results"


def main() -> None:
    seed = int(sys.argv[1])
    t0 = time.time()
    print(f"\n{'='*70}\n  ARM PF physfeat-cold seed {seed}  start "
          f"{time.strftime('%H:%M:%S')}\n{'='*70}", flush=True)
    try:
        run_bo(n_initial=20, n_iterations=80, q=1, seed=seed,
               log_dir=str(RESULTS / f"physfeat_cold_s{seed}"),
               use_physics=False, use_physics_features=True,
               use_warping=False, use_dkl=False,
               warmstart_centers=[PRESIM_X], resume_from=None,
               pid_tag="pfc", num_restarts=5, raw_samples=256)
        status = "OK"
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    print(f"\n  ARM PF seed {seed} {status} after {(time.time()-t0)/60:.1f} min",
          flush=True)
    print(f"LANE pf{seed} COMPLETE", flush=True)


if __name__ == "__main__":
    main()

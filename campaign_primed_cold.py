"""Pre-registered test: cheap-fidelity-primed COLD BO, one seed per lane.

Usage: campaign_primed_cold.py {42|7|11|23|101}

Each run: 20-point DoE (baseline-perturbed half, as Stage I) + 80 BO
iterations with the MLP cheap-fidelity priors (S11/S22/K/B1) and analytical
OIP3/Pdc/Gain priors, all under learnable per-output scaling. Controls: the
existing cold zero-mean runs (geo2 for seed 42, ms_stage1_s* for the rest).

Success criteria (fixed before launch): pooled BO-phase feasible rate vs the
zero-mean controls (one-sided Fisher); HV at the 100-eval budget under frozen
per-seed reference points; comparison of 100-eval primed HV vs the 180-eval
two-stage pipeline HV.
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
    print(f"\n{'='*70}\n  ARM PC primed-cold seed {seed}  start "
          f"{time.strftime('%H:%M:%S')}\n{'='*70}", flush=True)
    try:
        run_bo(n_initial=20, n_iterations=80, q=1, seed=seed,
               log_dir=str(RESULTS / f"primed_cold_s{seed}"),
               use_physics=True, use_warping=False, use_dkl=False,
               warmstart_centers=[PRESIM_X], resume_from=None,
               pid_tag="pc", num_restarts=5, raw_samples=256)
        status = "OK"
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    print(f"\n  ARM PC seed {seed} {status} after {(time.time()-t0)/60:.1f} min",
          flush=True)
    print(f"LANE pc{seed} COMPLETE", flush=True)


if __name__ == "__main__":
    main()

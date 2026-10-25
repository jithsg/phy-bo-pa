"""Extend the no-physics arm of the constraint ablation (zero-mean GP, no d_in
inequality, PDK snapping only) from 20 + 80 to 20 + 160 evaluations, so that
"standard BO with no physical information" exists at the same 180-evaluation
budget as the pipeline and the Stage-I sweep (KBS revision, R5-1 / R5-2 / R3-2).

    python3 campaign_kbs/cabl_ext_lane.py <seed>

Resumes results/cabl_nodin_s{seed}/all_results.json (100 points) into
results/cabl_nodin_ext_s{seed}/ with the paper settings (q=1, num_restarts=5,
raw_samples=256).  Point ids: cAx_s{seed}_*.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from config import LOCAL_RESULTS_DIR, PRESIM_X  # noqa: E402
from bo_loop_geo import run_bo  # noqa: E402

RESULTS = Path(LOCAL_RESULTS_DIR)
N_DOE = 20
N_TOTAL_ITERS = 160


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    seed = int(sys.argv[1])
    src = RESULTS / f"cabl_nodin_s{seed}" / "all_results.json"
    if not src.exists():
        raise FileNotFoundError(src)
    out = RESULTS / f"cabl_nodin_ext_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    own = out / "all_results.json"
    if not own.exists():
        pts = json.load(open(src))
        if len(pts) != N_DOE + 80:
            raise RuntimeError(f"{src} has {len(pts)} points, expected {N_DOE + 80}")
        shutil.copy(src, own)
    t0 = time.time()
    print(f"ARM no-d_in ext seed {seed} start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    try:
        run_bo(n_initial=N_DOE, n_iterations=N_TOTAL_ITERS, q=1, seed=seed,
               log_dir=str(out), resume_from=str(own), pid_tag="cAx",
               use_warping=False, use_dkl=False, num_restarts=5, raw_samples=256,
               use_physics=False, use_din_constraint=False,
               warmstart_centers=[PRESIM_X])
        status = "OK"
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    print(f"LANE cabl_ext_s{seed} {status} after {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()

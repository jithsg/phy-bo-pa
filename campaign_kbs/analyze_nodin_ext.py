"""Outcome of the no-physics arm (zero-mean GP, no d_in inequality, PDK snapping
only) extended to the common 180-evaluation budget, for the row that is added
beside the Stage-I sweep (manuscript Table 18 / Section 5.5; R5-1).

Hypervolume uses the per-seed reference point of Table 18 (computed over the
existing sweep conditions, exactly as analyze_kbs.analyze_sweep does), so the
new row is comparable with the published rows without changing them.

    python3 campaign_kbs/analyze_nodin_ext.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_kbs as A  # noqa: E402


def nodin_ext(seed: int) -> list[dict] | None:
    return A._load(A.RES / f"cabl_nodin_ext_s{seed}" / "all_results.json")


def main() -> None:
    print(f"{'seed':>5} {'n':>4} {'invalid':>7} {'feas':>5} {'bestPAE':>8} {'HV(T18 ref)':>12}  {'feas@100':>8} {'PAE@100':>8}")
    rows = {}
    for s in A.SEEDS:
        runs = {n1: A.sweep_run(s, n1) for n1 in A.N1_LIST}
        runs["paper"] = A.ours_run(s)
        avail = {k: r for k, r in runs.items() if r}
        Yall = np.vstack([A.objs_ours(r)[A.feas_ours(r)] for r in avail.values() if A.feas_ours(r).any()])
        ref = A.refpt(Yall)
        r = nodin_ext(s)
        if r is None:
            print(f"{s:>5}  missing")
            continue
        n = min(len(r), A.BUDGET)
        st = A.stats(A.objs_ours(r), A.feas_ours(r), ref, n)
        st100 = A.stats(A.objs_ours(r), A.feas_ours(r), ref, 100)
        invalid = sum(bool(p.get("rejected")) for p in r[A.N_DOE if hasattr(A, "N_DOE") else 20:n])
        flag = "" if len(r) >= A.BUDGET else f"  (partial: {len(r)}/{A.BUDGET})"
        print(f"{s:>5} {st['n']:>4} {invalid:>7} {st['feas']:>5} {st['best_pae']:>8.1f} {st['hv']:>12.0f}  "
              f"{st100['feas']:>8} {st100['best_pae']:>8.1f}{flag}")
        rows[s] = dict(st, invalid=invalid, feas100=st100["feas"], pae100=st100["best_pae"], complete=len(r) >= A.BUDGET)
    comp = [v for v in rows.values() if v["complete"]]
    if comp:
        print(f"\nmedian over {len(comp)} complete seeds: feas {np.median([v['feas'] for v in comp]):.0f}, "
              f"best PAE {np.nanmedian([v['best_pae'] for v in comp]):.1f}, HV {np.median([v['hv'] for v in comp]):.0f}, "
              f"invalid {np.median([v['invalid'] for v in comp]):.0f}")
    json.dump({str(k): v for k, v in rows.items()}, open(HERE.parent / "data" / "kbs" / "nodin_ext_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

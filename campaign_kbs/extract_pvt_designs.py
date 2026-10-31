"""Best feasible design (by PAE) of every optimizer and seed, with its design
vector, for the PVT corner campaign.  Only complete 180-evaluation runs are
used.  Writes campaign_kbs/pvt_designs.json (overwrites; pvt_campaign.py is
resumable so designs already simulated are skipped).

    python3 campaign_kbs/extract_pvt_designs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_kbs as A  # noqa: E402


def main() -> None:
    rows = []
    for s in A.SEEDS:
        o = A.ours_run(s)
        if o:
            F = [p for p in o[:A.BUDGET] if p.get("feasible") and not p.get("rejected")]
            b = max(F, key=lambda p: p["objectives"]["PAE_pct"])
            rows.append(dict(design_id=f"proposed_s{s}", method="proposed", seed=s,
                             pae=b["objectives"]["PAE_pct"], x=b["x"], pid=b["point_id"]))
        for m in A.BASELINES:
            bl = A.baseline_run(m, s)
            if not bl or len(bl) < A.BUDGET:
                continue
            F = [p for p in bl[:A.BUDGET] if p.get("is_feasible") and not p.get("rejected")]
            if not F:
                continue
            b = max(F, key=lambda p: p["pae_pct"])
            rows.append(dict(design_id=f"{m}_s{s}", method=m, seed=s, pae=b["pae_pct"],
                             x=b["x"], pid=b["point_id"]))
        p = A.pibo_run(s)
        if p and len(p) >= A.BUDGET:
            F = [q for q in p[:A.BUDGET] if q.get("feasible") and not q.get("rejected")]
            if F:
                b = max(F, key=lambda q: q["objectives"]["PAE_pct"])
                rows.append(dict(design_id=f"pibo_s{s}", method="pibo", seed=s,
                                 pae=b["objectives"]["PAE_pct"], x=b["x"], pid=b["point_id"]))
    json.dump(rows, open(HERE / "pvt_designs.json", "w"), indent=1)
    print(f"{len(rows)} designs: " + ", ".join(r["design_id"] for r in rows))


if __name__ == "__main__":
    main()

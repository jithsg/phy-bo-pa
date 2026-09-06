"""Extract OP1dB for all feasible designs + baseline from cached HB sweeps.

parser.parse_hb_sweep already returns the full Pin/Pout/Gain sweep arrays but
never reports the 1-dB compression point. This script computes OP1dB (output
power at the linearly interpolated crossing of gain = small-signal gain - 1 dB)
for the 17 feasible designs of the paper run (geo2 Stage I + geo21 Stage II)
and the manual baseline, entirely from cached raw data (no simulations).

Output: results/p1db_feasible.csv and a printed summary table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parser import parse_hb_sweep  # noqa: E402

BASE = Path(__file__).parent
RESULTS = BASE / "results"
RUN = RESULTS / "geo21_warm_geoaware_s42" / "all_results.json"
BASELINE_DIR = RESULTS / "baseline_recheck_2026"


def op1db_from_sweep(sweep: dict) -> float:
    """OP1dB (dBm): interpolated output power where gain drops 1 dB below SS."""
    pin = np.asarray(sweep["pin_dbm"], dtype=float)
    pout = np.asarray(sweep["pout_dbm"], dtype=float)
    gain = np.asarray(sweep["gain_db"], dtype=float)
    ok = ~np.isnan(gain)
    ss = np.nanmean(gain[ok][:10])
    target = ss - 1.0
    idx = np.where(ok & (gain < target))[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(pout[0])
    x0, x1 = gain[i - 1], gain[i]
    w = (x0 - target) / (x0 - x1) if x0 != x1 else 1.0
    return float(pout[i - 1] + w * (pout[i] - pout[i - 1]))


def main() -> None:
    rows = json.load(open(RUN))
    entries = [("baseline", "Manual baseline", BASELINE_DIR)]
    for i, e in enumerate(rows):
        if e.get("feasible"):
            stage = "I" if i < 100 else "II"
            entries.append((f"stage{stage}", e["point_id"], RESULTS / e["point_id"]))

    out = []
    for stage, name, d in entries:
        raw = d / "sim_main_raw"
        if not raw.exists():
            print(f"[skip] {name}: no sim_main_raw")
            continue
        sweep = parse_hb_sweep(str(raw))
        if "error" in sweep:
            print(f"[skip] {name}: {sweep['error']}")
            continue
        out.append({
            "stage": stage, "design": name,
            "OP1dB_dBm": op1db_from_sweep(sweep),
            "Gain_dB": sweep["Gain_dB"],
            "Psat_dBm": sweep["Psat_dBm"],
            "PAE_pct": sweep["PAE_pct"],
        })

    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "p1db_feasible.csv", index=False)
    print(df.round(2).to_string(index=False))
    fdf = df[df.stage != "baseline"]
    print(f"\nOP1dB over feasible designs: min={fdf.OP1dB_dBm.min():.2f}  "
          f"median={fdf.OP1dB_dBm.median():.2f}  max={fdf.OP1dB_dBm.max():.2f}  "
          f"(n={len(fdf)})")


if __name__ == "__main__":
    main()

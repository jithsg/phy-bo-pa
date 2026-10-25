"""Rows of manuscript Table 10 (corner verification of I-1, II-1 and the
reconstructed baseline) recomputed from pvt_results.csv, i.e. with the same
corner deck as Table 12, plus the per-corner ranges quoted in Section 4.7.

    python3 campaign_kbs/table10_from_csv.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = {"proposed_s42": "Best Stage I (I-1)", "paper_II1_s42": "Best Stage II (II-1)",
           "paper_baseline": "Baseline"}


def main() -> None:
    rows = list(csv.DictReader(open(HERE.parent / "data" / "kbs" / "pvt_results.csv")))
    for did, label in DESIGNS.items():
        r = [x for x in rows if x["design_id"] == did and not x.get("error")]
        if len(r) != 27:
            print(f"{label}: {len(r)}/27 conditions done")
            continue
        feas = sum(x["feasible"] == "1" for x in r)
        tt = [x for x in r if x["corner"] == "tt"]
        ff = [x for x in r if x["corner"] == "ff"]
        ss = [x for x in r if x["corner"] == "ss"]
        tt_feas = sum(x["feasible"] == "1" for x in tt)
        pae = [float(x["PAE_pct"]) for x in r]
        pdc = [float(x["Pdc_mW"]) for x in r]
        k = [float(x["K"]) for x in r]
        print(f"{label:22s} & {feas} & {tt_feas} & {min(pae):.1f} & {max(pdc):.1f} & {min(k):.2f} \\\\")
        print(f"   FF Pdc {min(float(x['Pdc_mW']) for x in ff):.1f}--{max(float(x['Pdc_mW']) for x in ff):.1f} mW; "
              f"SS Pdc {min(float(x['Pdc_mW']) for x in ss):.1f}--{max(float(x['Pdc_mW']) for x in ss):.1f} mW; "
              f"K at FF {min(float(x['K']) for x in ff):.2f}--{max(float(x['K']) for x in ff):.2f}; "
              f"nominal K {[float(x['K']) for x in tt if x['temp_c'] in ('27', '27.0') and float(x['vdd_v']) == 1.1][0]:.2f}; "
              f"nominal S11 {[float(x['S11_dB']) for x in tt if x['temp_c'] in ('27', '27.0') and float(x['vdd_v']) == 1.1][0]:.1f}; "
              f"fails by constraint: " + ", ".join(f"{c}={sum(c in x['fails'] for x in r)}" for c in ("S11_dB", "S22_dB", "Pdc_mW", "K", "OIP3_dBm", "B1")))
        print(f"   TT fails: " + "; ".join(f"{x['temp_c']}C/{x['vdd_v']}V:{x['fails']}" for x in tt if x["feasible"] != "1"))


if __name__ == "__main__":
    main()

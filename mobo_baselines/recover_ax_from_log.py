"""Reconstruct an Ax baseline trajectory JSON from the spectre campaign log
when the in-memory trajectory was lost (e.g. process killed before save).

Parses lines like:
    eval  098 [FEAS] PAE=57.79% Gain=16.89dB Psat=15.32dBm OIP3=22.1dBm Pdc=24.35mW S11=-15.32dB HV=18399.7281

and emits a JSON compatible with what `BaselineTrajectory.save_trajectory`
would have produced. Constraint metrics K, B1, S22, Vds_M0 are NOT in the
log; we fill them with conservative passing values when status == FEAS and
failing values when INFEAS — this is fine because feasibility is what
matters for the comparison.
"""

import argparse
import json
import re
from pathlib import Path

EVAL_RE = re.compile(
    r"eval\s+(\d+)\s+\[(FEAS|INFEAS|REJECT)\]\s+"
    r"PAE=(-?[\d.]+)%\s+Gain=(-?[\d.]+)dB\s+Psat=(-?[\d.]+)dBm\s+"
    r"OIP3=(-?[\d.]+)dBm\s+Pdc=(-?[\d.]+)mW\s+S11=(-?[\d.]+)dB\s+"
    r"HV=([\d.]+)"
)


def parse_segment(log_text: str, start_marker: str, end_marker: str | None = None):
    """Return the substring between start_marker and end_marker (or EOF)."""
    start = log_text.find(start_marker)
    if start < 0:
        raise ValueError(f"start marker not found: {start_marker!r}")
    after_start = log_text[start + len(start_marker):]
    if end_marker:
        end = after_start.find(end_marker)
        if end >= 0:
            after_start = after_start[:end]
    return after_start


def records_from_segment(segment: str, baseline_name: str, seed: int,
                         n_evals_cap: int) -> tuple[list, list]:
    records = []
    hv_history = []
    for line in segment.splitlines():
        m = EVAL_RE.search(line)
        if not m:
            continue
        idx, status, pae, gain, psat, oip3, pdc, s11, hv = m.groups()
        idx = int(idx)
        if idx >= n_evals_cap:
            break
        pae_f, gain_f, psat_f = float(pae), float(gain), float(psat)
        oip3_f, pdc_f, s11_f = float(oip3), float(pdc), float(s11)
        hv_f = float(hv)

        is_feasible = status == "FEAS"
        rejected = status == "REJECT"

        rec = {
            "point_id": f"baselines_{baseline_name}_s{seed}_eval{idx:04d}",
            "pae_pct": pae_f if not rejected else None,
            "gain_db": gain_f if not rejected else None,
            "psat_dbm": psat_f if not rejected else None,
            "K": None, "B1": None,  # not in log
            "S11_dB": s11_f if not rejected else None,
            "S22_dB": None,  # not in log
            "OIP3_dBm": oip3_f if not rejected else None,
            "Pdc_mW": pdc_f if not rejected else None,
            "is_feasible": is_feasible,
            "rejected": rejected,
            "sim_time_s": 0.0,  # not tracked here
        }
        records.append(rec)
        hv_history.append(hv_f)
    return records, hv_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--baseline", required=True,
                    help="e.g. ax_mobo or hebo_general")
    ap.add_argument("--start-marker", required=True,
                    help='e.g. ">>> ax_mobo (Spectre)"')
    ap.add_argument("--end-marker", default=None,
                    help='e.g. ">>> ax_mobo done in" — optional')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-evals-cap", type=int, default=180,
                    help="Truncate at this eval count (Stage-I = 100)")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    log_text = args.log.read_text()
    segment = parse_segment(log_text, args.start_marker, args.end_marker)
    records, hv_history = records_from_segment(
        segment, args.baseline, args.seed, args.n_evals_cap,
    )

    feasible_count = sum(1 for r in records if r["is_feasible"])
    payload = {
        "baseline_name": args.baseline,
        "seed": args.seed,
        "n_evals": len(records),
        "records": records,
        "hv_history": hv_history,
        "final_hv": hv_history[-1] if hv_history else 0.0,
        "feasible_count": feasible_count,
        "ref_point_obj": [0.0, 0.0, 0.0],
        "note": (
            f"Recovered from log (process was killed before save_trajectory). "
            f"Constraint metrics K/B1/S22 not in log — set to None. "
            f"Capped at {args.n_evals_cap} evals."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(records)} records ({feasible_count} feasible) "
          f"to {args.output}")
    if hv_history:
        print(f"Final HV: {hv_history[-1]:.4f}")


if __name__ == "__main__":
    main()

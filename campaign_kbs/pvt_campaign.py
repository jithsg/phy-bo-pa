"""PVT corner robustness of the best feasible design of every optimizer and
seed (KBS revision).  Uses the SAME evaluator as the campaigns (generated
decks, workstation Spectre via the slot wrapper, campaign parser) with the
process section, temperature and supply substituted, so the nominal
TT/27C/1.10V row reproduces the campaign record exactly.

    PA_RESULTS_DIR=<scratch> python3 campaign_kbs/pvt_campaign.py [--workers 8] [--smoke]

Input : campaign_kbs/pvt_designs.json   (from the best-design extraction)
Output: campaign_kbs/pvt_results.csv    (one row per design x condition; resumable)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

CORNERS = ("tt", "ss", "ff")          # the three corners used in the paper
TEMPS = (-40, 27, 85)
VDDS = (0.99, 1.10, 1.21)
SPECS = {"S11_dB": ("<=", -8.0), "S22_dB": ("<=", -8.0), "Pdc_mW": ("<=", 25.0),
         "K": (">=", 1.0), "B1": (">=", 0.0), "OIP3_dBm": (">=", 18.0)}
FIELDS = ["design_id", "method", "seed", "corner", "temp_c", "vdd_v", "PAE_pct", "Gain_dB",
          "Psat_dBm", "S11_dB", "S22_dB", "Pdc_mW", "K", "B1", "OIP3_dBm", "fails", "feasible",
          "sim_s", "error"]


def snap(x: list[float]) -> list[float]:
    return [round(x[0]), round(x[1] * 2) / 2, round(x[2]), round(x[3] * 2) / 2, x[4], x[5], x[6]]


def run_condition(design: dict, corner: str, temp: int, vdd: float) -> dict:
    """Runs in a worker process: generate, substitute, simulate, parse, clean up."""
    import parser as P                      # noqa: E402  (campaign parser)
    from netlist import generate_main_netlist, generate_oip3_netlist  # noqa: E402
    from simulator import run_design_point  # noqa: E402
    import config as C                      # noqa: E402

    x = snap(design["x"])
    pid = f"pvt_{design['design_id']}_{corner}_t{temp}_v{vdd:.2f}"
    row = dict(design_id=design["design_id"], method=design["method"], seed=design["seed"],
               corner=corner, temp_c=temp, vdd_v=vdd, fails="", feasible=0, error="")
    subs = [("section=top_tt", f"section=top_{corner}"),
            ("temp=27 tnom=27", f"temp={temp} tnom=27"),
            (f"vsource dc={C.VDD} type=dc", f"vsource dc={vdd} type=dc")]
    decks = []
    for gen in (generate_main_netlist, generate_oip3_netlist):
        nl = gen(*x)
        for old, new in subs:
            assert old in nl, f"substitution target missing: {old!r}"
            nl = nl.replace(old, new)
        decks.append(nl)
    t0 = time.time()
    sim = run_design_point(decks[0], decks[1], pid)
    row["sim_s"] = round(time.time() - t0, 1)
    try:
        if not sim["success"]:
            raise RuntimeError(sim.get("error") or "simulation failed")
        P.VDD = vdd                          # parser derives Pdc/PAE from VDD x I(V1)
        parsed = P.parse_all(sim["main_raw_dir"], sim["oip3_raw_dir"])
        obj, con = parsed["objectives"], parsed["constraints"]
        row.update(PAE_pct=obj["PAE_pct"], Gain_dB=obj["Gain_dB"], Psat_dBm=obj["Psat_dBm"],
                   S11_dB=con["S11_dB"], S22_dB=con["S22_dB"], Pdc_mW=con["Pdc_mW"],
                   K=con["K"], B1=con["B1"], OIP3_dBm=con["OIP3_dBm"])
        fails = [k for k, (op, lim) in SPECS.items()
                 if con.get(k) is None or not (con[k] <= lim if op == "<=" else con[k] >= lim)]
        row["fails"] = ";".join(fails)
        row["feasible"] = int(not fails)
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)[:120]
    finally:
        shutil.rmtree(os.path.join(C.LOCAL_RESULTS_DIR, pid), ignore_errors=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="one nominal condition of the first design")
    ap.add_argument("--only", nargs="*", default=None, help="design_id subset")
    args = ap.parse_args()
    designs = json.load(open(HERE.parent / "data" / "kbs" / "pvt_designs.json"))
    if args.only:
        designs = [d for d in designs if d["design_id"] in args.only]
    out = HERE.parent / "data" / "kbs" / "pvt_results.csv"
    done: set[tuple] = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            done.add((r["design_id"], r["corner"], int(r["temp_c"]), float(r["vdd_v"])))
    jobs = [(d, c, t, v) for d in designs for c, t, v in itertools.product(CORNERS, TEMPS, VDDS)
            if (d["design_id"], c, t, v) not in done]
    if args.smoke:
        jobs = [(designs[0], "tt", 27, 1.10)]
    print(f"{len(jobs)} conditions to run ({len(done)} already done), {args.workers} workers", flush=True)
    new_file = not out.exists()
    with open(out, "a", newline="") as f, ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        futs = {ex.submit(run_condition, *j): j for j in jobs}
        n = 0
        for fut in as_completed(futs):
            row = fut.result()
            w.writerow({k: row.get(k, "") for k in FIELDS}); f.flush()
            n += 1
            tag = f"{row['design_id']}/{row['corner']}/{row['temp_c']}C/{row['vdd_v']}V"
            if row["error"]:
                print(f"[{n}/{len(jobs)}] {tag} ERROR {row['error']}", flush=True)
            else:
                print(f"[{n}/{len(jobs)}] {tag} PAE={row['PAE_pct']:.1f} S11={row['S11_dB']:.1f} "
                      f"Pdc={row['Pdc_mW']:.1f} K={row['K']:.2f} feas={row['feasible']} "
                      f"fails={row['fails']} ({row['sim_s']}s)", flush=True)
    print("PVT CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()

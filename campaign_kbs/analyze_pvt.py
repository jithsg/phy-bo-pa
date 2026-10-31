"""Summarize the PVT corner campaign (campaign_kbs/pvt_results.csv): per design
feasible conditions /27, TT-only /9, worst-case metrics, and a per-seed
comparison of the proposed pipeline's design against every optimizer's.

    python3 campaign_kbs/analyze_pvt.py [--latex]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "pvt_results.csv"
METHODS = [("proposed", "Proposed"), ("hebo_general", "HEBO"), ("ax_mobo", "Ax qNEHVI"),
           ("optuna_motpe", "MOTPE"), ("pymoo_nsga2", "NSGA-II"), ("pymoo_nsga3", "NSGA-III"), ("pymoo_moead", "MOEA/D"),
           ("pibo", r"$\pi$BO-qLogNEHVI")]
SEEDS = ("42", "7", "11", "23", "101")


def load() -> dict[str, list[dict]]:
    rows = defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        if r["error"]:
            continue
        rows[r["design_id"]].append(r)
    return rows


def summarize(rs: list[dict]) -> dict:
    f = lambda k: np.array([float(r[k]) for r in rs])  # noqa: E731
    tt = [r for r in rs if r["corner"] == "tt"]
    nominal = [r for r in rs if r["corner"] == "tt" and r["temp_c"] == "27" and abs(float(r["vdd_v"]) - 1.10) < 1e-6]
    fails = defaultdict(int)
    for r in rs:
        for k in r["fails"].split(";"):
            if k:
                fails[k] += 1
    return dict(n=len(rs), feas=sum(int(r["feasible"]) for r in rs),
                tt_n=len(tt), tt_feas=sum(int(r["feasible"]) for r in tt),
                nom_pae=float(nominal[0]["PAE_pct"]) if nominal else np.nan,
                worst_pae=f("PAE_pct").min(), min_k=f("K").min(), max_pdc=f("Pdc_mW").max(),
                worst_s11=f("S11_dB").max(), worst_s22=f("S22_dB").max(), min_oip3=f("OIP3_dBm").min(),
                fails=dict(fails))


def main() -> None:
    latex = "--latex" in sys.argv
    data = load()
    summ = {d: summarize(rs) for d, rs in data.items()}
    print(f"{'design':<20}{'n':>4}{'feas/27':>8}{'TT/9':>6}{'nomPAE':>8}{'worstPAE':>9}{'minK':>6}{'maxPdc':>8}{'maxS11':>8}{'maxS22':>8}{'minOIP3':>8}  failing specs")
    for s in SEEDS:
        for key, label in METHODS:
            d = f"{key}_s{s}"
            if d not in summ:
                continue
            v = summ[d]
            fl = ",".join(f"{k}:{n}" for k, n in sorted(v["fails"].items(), key=lambda t: -t[1]))
            print(f"{d:<20}{v['n']:>4}{v['feas']:>8}{v['tt_feas']:>6}{v['nom_pae']:>8.1f}{v['worst_pae']:>9.1f}{v['min_k']:>6.2f}"
                  f"{v['max_pdc']:>8.1f}{v['worst_s11']:>8.1f}{v['worst_s22']:>8.1f}{v['min_oip3']:>8.1f}  {fl}")
    print("\nper-seed: proposed vs others on feasible conditions /27 (W = proposed strictly more robust)")
    for s in SEEDS:
        p = summ.get(f"proposed_s{s}")
        if not p or p["n"] < 27:
            continue
        others = {label: summ[f"{k}_s{s}"]["feas"] for k, label in METHODS[1:] if f"{k}_s{s}" in summ and summ[f"{k}_s{s}"]["n"] >= 27}
        verdict = "W" if others and all(p["feas"] > v for v in others.values()) else ("L" if others and any(v > p["feas"] for v in others.values()) else "T")
        print(f"  seed {s:>3}: proposed {p['feas']:>2}/27 | " + ", ".join(f"{k} {v}/27" for k, v in others.items()) + f"  -> {verdict}")
    if latex:
        print("\n% ---- LaTeX rows: seed & method & nominal PAE & feasible/27 & TT/9 & worst PAE & min K & max Pdc & worst S22 ----")
        for s in SEEDS:
            for key, label in METHODS:
                d = f"{key}_s{s}"
                if d in summ and summ[d]["n"] >= 27:
                    v = summ[d]
                    print(f"{s} & {label} & {v['nom_pae']:.1f} & {v['feas']} & {v['tt_feas']} & {v['worst_pae']:.1f} & {v['min_k']:.2f} & {v['max_pdc']:.1f} & {v['worst_s22']:.1f} \\\\")
            print(r"\midrule")


if __name__ == "__main__":
    main()

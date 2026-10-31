"""Emit LaTeX table bodies for the KBS revision from analysis_summary.json
(written by analyze_kbs.py).

    python3 campaign_kbs/make_tables.py            # prints both tables
    python3 campaign_kbs/make_tables.py sweep|baselines

Only complete runs (180 evaluations) are tabulated; partial runs are listed as
'--' so a draft can be compiled before the campaign finishes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "analysis_summary.json"
SEEDS = ("42", "7", "11", "23", "101")
N1_LIST = ("0", "20", "40", "60", "80", "100", "120", "160")
METHODS = [("proposed", "Proposed pipeline"), ("ax_mobo", "Ax qNEHVI"),
           ("hebo_general", "HEBO"), ("optuna_motpe", "Optuna MOTPE"),
           ("pymoo_nsga2", "NSGA-II"), ("pymoo_nsga3", "NSGA-III"), ("pymoo_moead", "MOEA/D"),
           ("pibo", r"$\pi$BO-qLogNEHVI")]


def _f(x, fmt="{:.1f}") -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def sweep_table(sw: dict) -> str:
    lines = [r"\begin{tabular}{@{}r" + "c" * len(SEEDS) + r"c@{}}", r"\toprule",
             r"$N_1$ & " + " & ".join(f"s{s}" for s in SEEDS) + r" & median \\", r"\midrule"]
    for metric, label, fmt in (("feas", "Feasible designs", "{:.0f}"),
                               ("best_pae", "Best feasible PAE (\\%)", "{:.1f}"),
                               ("hv", "Hypervolume", "{:,.0f}")):
        lines.append(r"\multicolumn{" + str(len(SEEDS) + 2) + r"}{@{}l}{\emph{" + label + r"}} \\")
        for n1 in N1_LIST:
            cells, vals = [], []
            for s in SEEDS:
                st = sw.get(s, {}).get(n1)
                if st and st.get("complete"):
                    cells.append(_f(st[metric], fmt)); vals.append(st[metric])
                else:
                    cells.append("--")
            med = _f(float(np.nanmedian(vals)), fmt) if vals else "--"
            lines.append(f"{n1} & " + " & ".join(cells) + f" & {med} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def baseline_table(bl: dict) -> str:
    """Table 11 body of the revised manuscript: HV relative to the best method of the seed."""
    lines = [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
             r"Seed & Method & Rej. & \makecell{Feas.\\@100} & \makecell{Feas.\\@180} & \makecell{PAE\\@100} & "
             r"\makecell{PAE\\@180} & \makecell{Gain\\@180} & \makecell{HV@180 /\\HV\textsubscript{best}} \\", r"\midrule"]
    for s in SEEDS:
        for key, label in METHODS:
            st = bl.get(s, {}).get(key)
            if not st:
                lines.append(f"{s} & {label} & -- & -- & -- & -- & -- & -- & -- \\\\")
                continue
            part = "" if st.get("complete") else r"\textsuperscript{p}"
            lines.append(f"{s} & {label}{part} & {st['rej']} & {st['feas100']} & {st['feas180']} & "
                         f"{_f(st['pae100'])} & {_f(st['pae180'])} & {_f(st.get('gain180'))} & {_f(st['hv_best'], '{:.2f}')} \\\\")
        lines.append(r"\midrule")
    # medians over seeds per method (complete runs)
    lines.append(r"\multicolumn{9}{@{}l}{\emph{Median over seeds (complete runs)}} \\")
    for key, label in METHODS:
        sts = [bl[s][key] for s in SEEDS if key in bl.get(s, {}) and bl[s][key].get("complete")]
        if not sts:
            continue
        med = lambda k: float(np.nanmedian([t[k] for t in sts]))  # noqa: E731
        lines.append(f"-- & {label} ({len(sts)}) & {med('rej'):.0f} & {med('feas100'):.0f} & {med('feas180'):.0f} & "
                     f"{med('pae100'):.1f} & {med('pae180'):.1f} & {med('gain180'):.1f} & {med('hv_best'):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = json.load(open(SUMMARY))
    if mode in ("sweep", "all") and "sweep" in d:
        print("% ---- Stage-I budget sweep ----")
        print(sweep_table(d["sweep"]))
    if mode in ("baselines", "all") and "baselines" in d:
        print("\n% ---- external optimizers ----")
        print(baseline_table(d["baselines"]))


if __name__ == "__main__":
    main()

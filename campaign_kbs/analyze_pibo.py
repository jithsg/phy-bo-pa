"""πBO-style baseline vs. the matched zero-mean control (and the proposed pipeline).

Per seed: evaluations, rejected proposals, Spectre count, feasible designs, best
feasible PAE (@100/@180), HV@180 with one reference point per seed over the union of
the feasible points of the three runs (πBO, zero-mean control, proposed pipeline),
optimizer time, Spectre time and their ratio.  Medians, paired seed-level differences
and exact two-sided sign tests.  Writes results/pibo/{pibo_results.csv, pibo_table.tex,
pibo_summary.json}.

Timing conventions (same as Table 13 of the manuscript):
  * Spectre time (normalised) = simulated proposals x 15.8 s (single-job median);
    the measured, contended per-evaluation times are reported separately.
  * Optimizer time = wall time between consecutive evaluations minus the logged
    simulation time, divided by the host speed factor relative to the May
    single-lane laptop (Mac mini 0.29 for this code; July 4-lane laptop 1.22;
    first WSL host 0.90, HEBO-based, used for the pc lanes).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "campaign_kbs"))
import analyze_kbs as K  # noqa: E402

RES, KBS, SEEDS, BUDGET, N_DOE = K.RES, K.KBS, K.SEEDS, K.BUDGET, K.N_DOE
SIM_REF_S = 15.8
FACTOR = {"mac": 0.29, "pc": 0.90, "pc3": 0.63, "laptop4": 1.22, "laptop": 1.0}
EXT_HOST = {42: "pc", 7: "mac", 11: "mac", 23: "mac", 101: "pc"}
OUT = RES / "pibo"
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------------ timing from logs
def iter_times(log: Path) -> dict[int, tuple[float, float]]:
    """{iteration: (optimizer_s, sim_s)} from a pipeline log ('Iteration time' lines)."""
    out, it, sim = {}, None, 0.0
    if not log.exists():
        return out
    for ln in open(log, errors="ignore"):
        m = re.search(r"Iter (\d+)/\d+", ln)
        if m:
            it, sim = int(m.group(1)), 0.0
        m = re.search(r"\((\d+\.\d)s\)\s*$", ln)
        if m and it is not None:
            sim = float(m.group(1))
        m = re.search(r"Iteration time: (\d+\.\d)s", ln)
        if m and it is not None:
            out[it] = (max(float(m.group(1)) - sim, 0.0), sim)
            it = None
    return out


def pibo_timing(seed: int) -> dict:
    t = iter_times(RES / f"pibo_s{seed}" / f"pibo_s{seed}.log")
    opt = sum(v[0] for v in t.values())
    sim = sum(v[1] for v in t.values())
    return {"iters_logged": len(t), "opt_meas_min": opt / 60, "sim_meas_min": sim / 60,
            "opt_eq_min": opt / FACTOR["mac"] / 60,
            "opt_per_iter_meas_s": float(np.median([v[0] for v in t.values()])) if t else float("nan"),
            "sim_per_eval_meas_s": float(np.median([v[1] for v in t.values()])) if t else float("nan")}


def control_timing(seed: int) -> dict:
    """Zero-mean control: iterations 1-80 from the July 4-lane laptop logs (seed 42: no
    log survives for cabl_din_s42 -> median of the other seeds, flagged), 81-160 from the
    ext lane on its campaign host."""
    first = iter_times(BASE / f"campaign_lane_s{seed}.log")
    first = {i: v for i, v in first.items() if i <= 80}
    ext = iter_times(KBS / f"ext_s{seed}" / f"ext_s{seed}.log")
    out = {"first80_eq_min": sum(v[0] for v in first.values()) / FACTOR["laptop4"] / 60 if first else float("nan"),
           "first80_n": len(first),
           "ext_eq_min": sum(v[0] for v in ext.values()) / FACTOR[EXT_HOST[seed]] / 60,
           "ext_n": len(ext), "ext_host": EXT_HOST[seed]}
    return out


# ------------------------------------------------------------------------ metrics
def run_metrics(recs: list[dict], ref: np.ndarray) -> dict:
    Y, F = K.objs_ours(recs), K.feas_ours(recs)
    rej = sum(1 for r in recs if r.get("rejected"))
    a, b = K.stats(Y, F, ref, 100), K.stats(Y, F, ref, BUDGET)
    sim_meas = sum(float(r.get("sim_time_s") or 0.0) for r in recs[N_DOE:])
    return {"n": len(recs), "rej": rej, "spectre_new": max(len(recs) - N_DOE - rej, 0),
            "feas100": a["feas"], "feas180": b["feas"], "pae100": a["best_pae"], "pae180": b["best_pae"],
            "hv180": b["hv"], "sim_meas_min": sim_meas / 60, "complete": len(recs) >= BUDGET}


def sign_test(d: list[float]) -> tuple[int, int, int, float]:
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    tie = len(d) - pos - neg
    p = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
    return pos, neg, tie, p


def main() -> None:
    rows, per_seed = [], {}
    print(f"{'seed':>4} {'run':<9} {'n':>4} {'rej':>3} {'feas':>4} {'PAE100':>6} {'PAE180':>6} {'HV180':>8} {'HV/ctrl':>7}")
    for s in SEEDS:
        runs = {"pibo": K._load(RES / f"pibo_s{s}" / "all_results.json"),
                "control": K.sweep_run(s, 160), "pipeline": K.ours_run(s)}
        runs = {k: v for k, v in runs.items() if v}
        if "pibo" not in runs or "control" not in runs:
            print(f"{s:>4} missing: {[k for k in ('pibo', 'control', 'pipeline') if k not in runs]}")
            continue
        Yall = np.vstack([K.objs_ours(r)[K.feas_ours(r)] for r in runs.values() if K.feas_ours(r).any()]
                         or [np.zeros((1, 3))])
        ref = K.refpt(Yall)
        m = {k: run_metrics(r, ref) for k, r in runs.items()}
        hv_c = m["control"]["hv180"]
        for k in m:
            m[k]["hv_rel_control"] = m[k]["hv180"] / hv_c if hv_c > 0 else float("nan")
        m["pibo"].update({f"t_{a}": b for a, b in pibo_timing(s).items()})
        m["pibo"]["spectre_eq_min"] = m["pibo"]["spectre_new"] * SIM_REF_S / 60
        m["pibo"]["ratio_opt_spectre"] = m["pibo"]["t_opt_eq_min"] / m["pibo"]["spectre_eq_min"] if m["pibo"]["spectre_eq_min"] else float("nan")
        ct = control_timing(s)
        m["control"].update({f"t_{a}": b for a, b in ct.items()})
        m["control"]["spectre_eq_min"] = m["control"]["spectre_new"] * SIM_REF_S / 60
        per_seed[s] = m
        for k, v in m.items():
            flag = "" if v["complete"] else f"  partial {v['n']}/{BUDGET}"
            print(f"{s:>4} {k:<9} {v['n']:>4} {v['rej']:>3} {v['feas180']:>4} {v['pae100']:>6.1f} {v['pae180']:>6.1f} "
                  f"{v['hv180']:>8.0f} {v['hv_rel_control']:>7.2f}{flag}")
            rows.append({"seed": s, "run": k, **{a: b for a, b in v.items()}})
    if not per_seed:
        return
    # ---- control optimizer time: fill seed 42's missing first-80 block with the median of the others
    f80 = [per_seed[s]["control"]["t_first80_eq_min"] for s in per_seed if per_seed[s]["control"]["t_first80_n"] >= 70]
    for s in per_seed:
        c = per_seed[s]["control"]
        if c["t_first80_n"] < 70 and f80:
            c["t_first80_eq_min"], c["t_first80_estimated"] = float(np.median(f80)), True
        else:
            c["t_first80_estimated"] = False
        c["t_opt_eq_min"] = c["t_first80_eq_min"] + c["t_ext_eq_min"]
        c["ratio_opt_spectre"] = c["t_opt_eq_min"] / c["spectre_eq_min"] if c["spectre_eq_min"] else float("nan")

    # ---- paired differences and sign tests (complete seeds only)
    done = [s for s in per_seed if per_seed[s]["pibo"]["complete"] and per_seed[s]["control"]["complete"]]
    tests = {}
    print(f"\npaired πBO − control over {len(done)} complete seeds:")
    for key, lab in (("pae180", "best feasible PAE@180"), ("hv180", "HV@180"), ("feas180", "feasible designs@180"),
                     ("pae100", "best feasible PAE@100")):
        d = []
        for s in done:
            a, b = per_seed[s]["pibo"][key], per_seed[s]["control"][key]
            a = -1e9 if a != a else a
            b = -1e9 if b != b else b
            d.append(a - b)
        pos, neg, tie, p = sign_test(d)
        tests[key] = {"diffs": d, "higher": pos, "lower": neg, "tied": tie, "p_sign": p,
                      "median_pibo": float(np.nanmedian([per_seed[s]["pibo"][key] for s in done])),
                      "median_control": float(np.nanmedian([per_seed[s]["control"][key] for s in done]))}
        print(f"  {lab:<26} higher/lower/tied = {pos}/{neg}/{tie}  p = {p:.3f}  "
              f"median πBO {tests[key]['median_pibo']:.2f} vs control {tests[key]['median_control']:.2f}  diffs {np.round(d, 2).tolist()}")

    # ---- CSV
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "pibo_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- LaTeX table
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{$\pi$BO-Style Prior-Weighted qLogNEHVI versus the Matched Zero-Mean Control (180 Evaluations, Five Seeds)}",
         r"\label{tab:pibo}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Seed & Method & Rej. & Feas.@180 & PAE@100 & PAE@180 & HV@180 / HV\textsubscript{ctrl} & $t_\mathrm{opt}/t_\mathrm{Spectre}$ \\",
         r"\midrule"]
    lab = {"pibo": r"$\pi$BO-qLogNEHVI", "control": "Zero-mean control", "pipeline": "Proposed pipeline"}

    def fmt(v, nd=1):
        return "--" if v != v else f"{v:.{nd}f}"

    for s in SEEDS:
        if s not in per_seed:
            continue
        for k in ("pibo", "control", "pipeline"):
            if k not in per_seed[s]:
                continue
            v = per_seed[s][k]
            ratio = fmt(v.get("ratio_opt_spectre", float("nan")), 2) if k != "pipeline" else "1.2"
            L.append(f"{s} & {lab[k]} & {v['rej']} & {v['feas180']} & {fmt(v['pae100'])} & {fmt(v['pae180'])} & "
                     f"{fmt(v['hv_rel_control'], 2)} & {ratio} \\\\")
    L += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Median over seeds}} \\"]
    for k in ("pibo", "control", "pipeline"):
        vals = [per_seed[s][k] for s in done if k in per_seed[s]]
        if not vals:
            continue
        med = lambda key, nd=1: fmt(float(np.nanmedian([v[key] for v in vals])), nd)  # noqa: E731
        ratio = med("ratio_opt_spectre", 2) if k != "pipeline" else "1.2"
        L.append(f"-- & {lab[k]} ({len(vals)}) & {med('rej', 0)} & {med('feas180', 0)} & {med('pae100')} & {med('pae180')} & "
                 f"{med('hv_rel_control', 2)} & {ratio} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "pibo_table.tex").write_text("\n".join(L) + "\n")
    json.dump({"per_seed": per_seed, "tests": tests, "complete_seeds": done}, open(OUT / "pibo_summary.json", "w"),
              indent=1, default=lambda x: None if x != x else x)
    print(f"\nwritten {OUT}/pibo_results.csv, pibo_table.tex, pibo_summary.json")


if __name__ == "__main__":
    main()

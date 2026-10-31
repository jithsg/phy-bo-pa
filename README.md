# phy-bo-pa — Physics-Informed Bayesian Optimization of CMOS Power Amplifiers

Code and per-evaluation data for the paper:

> *Geometry-Aware Multi-Objective Bayesian Optimization for CMOS Power Amplifier
> Design: Evaluating Physical Knowledge Integration* — J. Jithish,
> J. S. Ang, N. Mahalingam, B. K. Thangarasu, K. S. Yeo (Knowledge-Based
> Systems, under review; revised October 2026. Earlier title: *Physics-Informed
> Bayesian Optimization of CMOS Power Amplifiers: Constraints, Representations,
> and Priors*).

The paper compares three mechanisms for injecting physics knowledge into
multi-objective Bayesian optimization (qLogNEHVI/BoTorch) of a 2.4 GHz
push-pull class-AB PA in a 40 nm CMOS PDK: an analytical geometry
**constraint**, physics-derived kernel **representations**, and
physics-informed GP mean **priors**.

## What is (and is not) in this repository

**Included** — the full optimization framework (BO loops, physics models,
netlist generator, result parsers, campaign drivers, baselines, analysis
scripts) and the per-evaluation logs of every campaign reported in the paper
(design variables, objectives, constraint values, feasibility, per-run
configs, hypervolume histories), plus the S-parameter dataset and trained
surrogate used for the learned-prior experiment.

**Withheld under the foundry NDA** — PDK-specific values: the spiral
inductor's fixed trace width, turn spacing, metal layer and ground distance,
the design-rule minimum inner diameter, MOS-cap ring counts, the foundry
characterization table, and the PDK model files themselves. Derived
quantities that would reveal them (per-design inner diameters) are stripped
from the logs, including the October 2026 revision campaigns under
`data/runs/kbs_results/` and `data/runs/cabl_nodin_ext_s*/`. To *re-run simulations* you need access to the TSMC 40 nm PDK
and Cadence Spectre: copy `pdk_private.example.py` to `pdk_private.py` and
fill in the values from your PDK documentation.

**All paper tables and statistics reproduce from the shipped data alone —
no PDK required.**

## Layout

```
config.py                   public configuration (search space, thresholds)
pdk_private.example.py      template for NDA'd values (simulation only)
bo_loop.py                  Stage-I BO loop (non-geometry-aware)
bo_loop_geo.py              geometry-aware BO loop (snap + d_in inequality,
                            physics means, physics features; all ablation flags)
physics_models_geo.py       Wheeler/class-AB physics priors (calibrated K1/K2)
physics_sparam.py           physics-derived kernel features
mlp_prior.py, train_sparam_prior.py   learned S-parameter prior
netlist.py, evaluate.py, parser.py, simulator.py   Spectre evaluation chain
campaign_*.py, run_*.py     campaign drivers for every experiment
analyze_constraint_ablation.py, seedlevel_repr_table.py   table generators
mobo_baselines/             Ax / HEBO / Optuna / pymoo baselines (NSGA-II, NSGA-III, MOEA/D)
pibo_prior.py, pibo_lane.py, pibo_validate.py   piBO-style prior-weighted qLogNEHVI baseline
                            (Hvarfner et al. 2022 rule on the zero-mean backbone; prior centred on
                            the manual design; validation checks) -- Section 4.9, Tables 11-13
sweep_lane.py               Stage-I budget sweep: extend the zero-mean stage and branch the
                            physics-informed stage at N1 (Section 5.5)
campaign_kbs/               revision campaign: analysis of the five-seed benchmark and sweep
                            (analyze_kbs.py, make_tables.py), corner re-simulation of every
                            optimizer's best design (extract_pvt_designs.py, pvt_campaign.py,
                            analyze_pvt.py, table10_from_csv.py), no-physics extension
                            (cabl_ext_lane.py, analyze_nodin_ext.py)
plots/                      figure generators for Figs. 10-13 (Pareto sets, convergence, sweep)
data/runs/<campaign>/       per-evaluation logs (all_results.json, config.json,
                            hv_history.json)
data/sp_batch.csv           S-parameter dataset for the learned prior
data/sparam_mlp.pt          trained surrogate (R² 0.73/0.80 for S11/S22)
data/random_search/         budget-matched random-search control
data/pvt/pvt_summary.csv    corner-verification results of the original study
data/runs/kbs_results/      revision campaign: bl_<method>_s<seed>/ (five-seed benchmark,
                            Table 11), br_n<N1>_s<seed>/ and ext_s<seed>/ (Stage-I budget
                            sweep, Table 18), JSON only
data/runs/kbs_results/pibo_s<seed>/   piBO-style baseline, five seeds (Table 11, 8th method)
data/kbs/pibo/              its per-seed CSV, validation output, timing replay and the
                            Spectre/optimizer cost breakdown of every method (Table 13)
data/runs/cabl_nodin_ext_s* no-physics arm continued to 180 evaluations (Table 18, ‡ row)
data/runs/mobo_baselines/   seed-42 external-optimizer runs of the original study
data/kbs/pvt_results.csv    27-condition corner check of 41 designs (Tables 10 and 12)
data/kbs/analysis_summary.json, nodin_ext_summary.json   outputs of the analysis scripts
data/constraint_ablation_summary.csv
```

## Mapping paper results to data

| Paper item | Data / script |
|---|---|
| Stage I / Stage II feasible designs, Pareto, HV | `data/runs/geo2_warm_s42`, `data/runs/geo21_warm_geoaware_s42` |
| Factorized ablation (a) | above + `data/runs/ablA_warmonly_s42`, `data/runs/ablB_physscratch_s42` |
| Five-seed two-stage replication (b) | `data/runs/ms_stage1_s*`, `data/runs/ms_stage2_s*` |
| Learned-prior test (c) | `data/runs/primed_cold_s*` vs. controls |
| Representation test | `data/runs/physfeat_cold_s*` + `seedlevel_repr_table.py` |
| Constraint-only ablation | `data/runs/cabl_*` + `analyze_constraint_ablation.py` |
| Random-search control | `data/random_search/` |
| Corner verification, Table 10 and Section 4.7 | `data/kbs/pvt_results.csv` + `campaign_kbs/table10_from_csv.py` (designs `proposed_s42`, `paper_II1_s42`, `paper_baseline`) |
| Benchmark against seven optimizers, Table 11, Figs. 10-11 | `data/runs/kbs_results/bl_*` (+ `data/runs/mobo_baselines/*_seed42.json`) + `campaign_kbs/analyze_kbs.py all`, `campaign_kbs/make_tables.py`, `plots/generate_pareto3d.py`, `plots/generate_convergence.py` |
| piBO-style prior-weighted baseline vs. its zero-mean control (Table 11 row, Section 4.9) | `data/runs/kbs_results/pibo_s*` vs. `ext_s*` + `campaign_kbs/analyze_pibo.py`; validation `pibo_validate.py 42`; run `pibo_lane.py <seed>` |
| Spectre / optimizer / total time per method, Table 13 | `data/kbs/pibo/cost_breakdown.csv` (`campaign_kbs/replay_timing.py` for the host factor; lane logs are not shipped) |
| Corner robustness of every optimizer's best design, Table 12 | `data/kbs/pvt_results.csv` + `campaign_kbs/analyze_pvt.py --latex` |
| Stage-I budget sweep, Table 18, Figs. 12-13 | `data/runs/kbs_results/br_n*_s*`, `ext_s*` + `campaign_kbs/analyze_kbs.py all`, `plots/generate_sweep_fig.py`, `plots/generate_sweep_traces.py` |
| No-physics arm at 180 evaluations (Table 18, ‡ row) | `data/runs/cabl_nodin_ext_s*` + `campaign_kbs/analyze_nodin_ext.py` |

Each `all_results.json` record: 7-D design vector `x` (`[r_in, n_in, r_drn,
n_drn, C_wr(m), R_l(m), W_mos(m)]`), snapped inductor geometry (`rad`, `nr`),
objectives (PAE %, Gain dB, Psat dBm), six constraint values, feasibility,
simulation time. Geometrically rejected proposals carry `"rejected": true`
and consumed no simulation.

## Environment

Python 3.10, `torch` 2.11, `botorch` 0.16, `gpytorch` 1.15 (see
`requirements.txt`). External optimizers: `ax-platform` 1.2.1, `HEBO` 0.3.6,
`optuna` 4.8.0, `pymoo` 0.6.0. Simulations used Cadence Spectre 20.10.068 with the
TSMC 40 nm PDK (TT corner; corners TT/SS/FF for the PVT study).

## License

Code and data are released under the MIT License (see `LICENSE`). The PDK,
its model files, and foundry characterization data are not included and are
not covered by this license.

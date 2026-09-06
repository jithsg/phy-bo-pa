# phy-bo-pa — Physics-Informed Bayesian Optimization of CMOS Power Amplifiers

Code and per-evaluation data for the paper:

> *Physics-Informed Bayesian Optimization of CMOS Power Amplifiers:
> Constraints, Representations, and Priors* — J. Jithish, J. S. Ang,
> N. Mahalingam, B. K. Thangarasu, K. S. Yeo (under review).

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
from the logs. To *re-run simulations* you need access to the TSMC 40 nm PDK
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
mobo_baselines/             Ax / HEBO / Optuna / pymoo baselines
data/runs/<campaign>/       per-evaluation logs (all_results.json, config.json,
                            hv_history.json)
data/sp_batch.csv           S-parameter dataset for the learned prior
data/sparam_mlp.pt          trained surrogate (R² 0.73/0.80 for S11/S22)
data/random_search/         budget-matched random-search control
data/pvt/pvt_summary.csv    corner-verification results (27 conditions/design)
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
| Corner verification | `data/pvt/pvt_summary.csv` |

Each `all_results.json` record: 7-D design vector `x` (`[r_in, n_in, r_drn,
n_drn, C_wr(m), R_l(m), W_mos(m)]`), snapped inductor geometry (`rad`, `nr`),
objectives (PAE %, Gain dB, Psat dBm), six constraint values, feasibility,
simulation time. Geometrically rejected proposals carry `"rejected": true`
and consumed no simulation.

## Environment

Python 3.10, `torch` 2.11, `botorch` 0.16, `gpytorch` 1.15 (see
`requirements.txt`). Simulations used Cadence Spectre 20.10.068 with the
TSMC 40 nm PDK (TT corner; corners TT/SS/FF for the PVT study).

## License

Code and data are released under the MIT License (see `LICENSE`). The PDK,
its model files, and foundry characterization data are not included and are
not covered by this license.

"""Generate fixed random-search sample sets for the RS baseline.

Matches the paper's experimental protocol:
  - 5 seeds (42, 7, 11, 23, 101), 180 evaluations each (same budget as
    Stage I + Stage II combined).
  - Uniform sampling over the 7-D design box of Table II.
  - PDK snapping identical to the BO pipeline: radius -> integer um,
    turn count -> half-turn increments (applied before simulation; the
    raw continuous sample is also recorded).
  - No inner-diameter screening: random search plays the role of the
    unconstrained null, so invalid geometries receive the same
    conservative infeasible values as failed simulations in the paper.

Outputs one CSV per seed: rs_samples_seed<seed>.csv
"""

import csv
import numpy as np

SEEDS = [42, 7, 11, 23, 101]
N_EVALS = 180

# Table II bounds: (name, min, max)
BOUNDS = [
    ("r_in_um",   30.0, 45.0),    # gate-side spiral radius
    ("n_in",      4.5,  7.0),     # gate-side turn count
    ("r_drn_um",  30.0, 45.0),    # drain-side spiral radius
    ("n_drn",     3.5,  6.0),     # drain-side turn count
    ("c_wr_fF",   138.0, 1171.0), # match capacitor
    ("r_l_ohm",   50.0, 503.0),   # bias resistor
    ("w_mos_um",  1.50, 5.00),    # MOSFET width per finger (multi=100)
]


def snap(row):
    """PDK grid snapping: r -> integer um, n -> half-turns."""
    out = dict(row)
    out["r_in_um"] = round(row["r_in_um"])
    out["r_drn_um"] = round(row["r_drn_um"])
    out["n_in"] = round(2.0 * row["n_in"]) / 2.0
    out["n_drn"] = round(2.0 * row["n_drn"]) / 2.0
    return out


def main():
    names = [b[0] for b in BOUNDS]
    lo = np.array([b[1] for b in BOUNDS])
    hi = np.array([b[2] for b in BOUNDS])

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        raw = lo + (hi - lo) * rng.random((N_EVALS, len(BOUNDS)))
        path = f"rs_samples_seed{seed}.csv"
        with open(path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(
                ["idx"]
                + [f"raw_{n}" for n in names]
                + [f"snap_{n}" for n in names]
            )
            for i, x in enumerate(raw):
                row = dict(zip(names, x))
                s = snap(row)
                wtr.writerow(
                    [i]
                    + [f"{row[n]:.6g}" for n in names]
                    + [f"{s[n]:.6g}" for n in names]
                )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

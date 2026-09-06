"""Aggregate the four baseline trajectories alongside the existing
``geo2`` (Base-BO) and ``geo21`` (Phys-BO) runs and write a comparison
summary to ``results/mobo_baselines/comparison_spectre/``.

Outputs:
  * ``summary.md`` — markdown table (best PAE / Gain / Psat / FOM /
    feasibility / wall time, with snapshots at evaluations 100 and 180)
  * ``trajectories.json`` — combined HV-vs-eval data for all methods
  * ``hv_convergence.png`` — HV trajectory plot (optional, only if matplotlib
    is available)
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _read_baseline(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return d


def _best_metrics_at(records: list[dict], n: int) -> dict:
    seen = records[:n]
    feas = [r for r in seen if r.get("is_feasible")]
    if not feas:
        return {
            "best_pae": None, "best_gain": None, "best_psat": None,
            "n_feas": 0, "n_evals": len(seen),
        }
    return {
        "best_pae":  max(r["pae_pct"]  for r in feas if r["pae_pct"]  is not None),
        "best_gain": max(r["gain_db"]  for r in feas if r["gain_db"]  is not None),
        "best_psat": max(r["psat_dbm"] for r in feas if r["psat_dbm"] is not None),
        "n_feas":    len(feas),
        "n_evals":   len(seen),
    }


def _read_proposed_runs(project_root: Path) -> list[dict]:
    """Pull whatever Base-BO / Phys-BO runs exist in the parent results dir.

    We look for the canonical ``geo2_warm_s42`` (Base-BO Stage I) and
    ``geo21*`` (Phys-BO Stage II) ``all_results.json`` files. If a run is
    missing we skip it silently — the comparison still shows what's present.
    """
    proposed = []
    for slug in ("geo2_warm_s42", "geo21_warm_geoaware_s42",
                 "geo21_warm_s42", "geo20_warm_geoaware_s42"):
        path = project_root / "results" / slug / "all_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        records = []
        for r in raw:
            obj = r.get("objectives", {}) or {}
            con = r.get("constraints", {}) or {}
            rejected = bool(r.get("rejected", False))
            from .common import constraint_margins, is_feasible
            margins = constraint_margins(con) if not rejected else {
                "c_K": -1, "c_B1": -1, "c_S11": -1, "c_S22": -1,
                "c_OIP3": -1, "c_Pdc": -1,
            }
            records.append({
                "pae_pct":  obj.get("PAE_pct"),
                "gain_db":  obj.get("Gain_dB"),
                "psat_dbm": obj.get("Psat_dBm"),
                "is_feasible": (not rejected) and is_feasible(margins),
                "sim_time_s": r.get("sim_time_s", 0.0),
            })
        proposed.append({
            "baseline_name": slug,
            "records": records,
            "n_evals": len(records),
        })
    return proposed


def write_comparison(baselines_dir: Path, output_dir: Path,
                     project_root: Optional[Path] = None) -> Path:
    project_root = project_root or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods: list[dict] = []
    for json_path in sorted(baselines_dir.glob("*.json")):
        methods.append(_read_baseline(json_path))

    proposed = _read_proposed_runs(project_root)
    methods.extend(proposed)

    if not methods:
        logger.warning("No trajectories to aggregate.")
        return output_dir

    # === Markdown summary at eval-100 and eval-180 snapshots ===
    lines = [
        "# PA Baselines vs Proposed — Spectre comparison",
        "",
        "Each method is reported at two snapshot budgets:",
        "**eval 100** (matches Stage I `geo2` budget) and",
        "**eval 180** (matches total Stage I + Stage II budget).",
        "",
        "| Method | n_feas@100 | best PAE@100 (%) | best Gain@100 (dB) | best Psat@100 (dBm) "
        "| n_feas@180 | best PAE@180 (%) | best Gain@180 (dB) | best Psat@180 (dBm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in methods:
        recs = m["records"]
        a = _best_metrics_at(recs, 100)
        b = _best_metrics_at(recs, 180)
        def fmt(v, prec=2):
            return f"{v:.{prec}f}" if v is not None else "n/a"
        lines.append(
            f"| {m['baseline_name']} | {a['n_feas']} | {fmt(a['best_pae'])} | "
            f"{fmt(a['best_gain'])} | {fmt(a['best_psat'])} | "
            f"{b['n_feas']} | {fmt(b['best_pae'])} | "
            f"{fmt(b['best_gain'])} | {fmt(b['best_psat'])} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", output_dir / "summary.md")

    # === Combined trajectory JSON ===
    combined = []
    for m in methods:
        recs = m["records"]
        combined.append({
            "name": m["baseline_name"],
            "n_evals": len(recs),
            "hv_history": m.get("hv_history"),  # only for baselines
            "feasible_count_at": {
                "100": _best_metrics_at(recs, 100)["n_feas"],
                "180": _best_metrics_at(recs, 180)["n_feas"],
            },
        })
    with open(output_dir / "trajectories.json", "w") as f:
        json.dump(combined, f, indent=2)
    logger.info("Wrote %s", output_dir / "trajectories.json")

    # === Optional HV plot ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for m in methods:
            hv = m.get("hv_history")
            if not hv:
                continue
            ax.plot(range(1, len(hv) + 1), hv, label=m["baseline_name"],
                    linewidth=1.4, alpha=0.9)
        ax.axvline(100, color="gray", linestyle=":", linewidth=0.8)
        ax.text(100.5, ax.get_ylim()[1] * 0.05, "eval 100",
                color="gray", fontsize=8)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Hypervolume (feasible)")
        ax.set_title("PA baseline HV trajectories (Spectre)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "hv_convergence.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s", output_dir / "hv_convergence.png")
    except ImportError:
        pass

    return output_dir


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baselines-dir", type=Path,
        default=Path("results/mobo_baselines"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("results/mobo_baselines/comparison_spectre"),
    )
    args = p.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    write_comparison(args.baselines_dir, args.output_dir)


if __name__ == "__main__":
    main()

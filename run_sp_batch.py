"""Cheap-fidelity S-parameter batch: ~1,000 SP-only simulations.

Generates Sobol designs over the full bounds (geometry-filtered for d_in
validity), strips the 200-point HB sweep from the main netlist (keeping
DC + SP), and runs them on the workstation in chunks of 10 per SSH
round-trip. Each design yields S11_dB, S22_dB, K, B1 at ~2 s instead of the
~15 s full evaluation — training data for cheap-fidelity constraint priors.

Restartable: completed rows live in results/sp_batch.csv keyed by design
index; re-running skips them. Remote chunk dirs are deleted after fetch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from bo_loop_geo import snap_to_pdk_grid
from config import BOUNDS_LOWER, BOUNDS_UPPER, check_d_in, IND_W, IND_SPACING
from netlist import generate_main_netlist
from parser import parse_sp
from simulator import _sim_lock, SPECTRE_BIN

REMOTE_BASE = "~/bo_runs/sp_batch"
OUT_CSV = BASE / "results" / "sp_batch.csv"
N_TARGET = 1000
CHUNK = 10
COLS = ["idx", "L1L2_rad", "L1L2_nr", "L3L4_rad", "L3L4_nr",
        "c_wr", "r_l", "m_wr", "S11_dB", "S22_dB", "K", "B1", "error"]

_HB_BLOCK = re.compile(r"sweephb sweep.*?\n\}\n", re.S)


def sp_netlist(x: np.ndarray) -> str:
    """Main netlist with the HB sweep stripped (DC + SP only)."""
    full = generate_main_netlist(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
    stripped, n = _HB_BLOCK.subn("", full)
    assert n == 1, "HB sweep block not found/removed"
    return stripped


def sobol_designs(n: int) -> np.ndarray:
    """Geometry-valid Sobol designs in raw units."""
    eng = torch.quasirandom.SobolEngine(7, scramble=True, seed=2026)
    out, seen = [], set()
    while len(out) < n:
        u = eng.draw(256).double().numpy()
        x = BOUNDS_LOWER + u * (BOUNDS_UPPER - BOUNDS_LOWER)
        # PDK cells require grid-valid inductor geometry: snap BEFORE
        # simulation (the same snap every real evaluation applies), and
        # dedupe designs that collapse onto the same grid point.
        x = snap_to_pdk_grid(x)
        for row in x:
            key = tuple(np.round(row, 9))
            if key in seen:
                continue
            if (check_d_in(row[0], row[1], IND_W, IND_SPACING)
                    and check_d_in(row[2], row[3], IND_W, IND_SPACING)):
                seen.add(key)
                out.append(row)
                if len(out) == n:
                    break
    return np.array(out)


def run_chunk(designs: list[tuple[int, np.ndarray]]) -> list[dict]:
    """Upload, simulate, fetch, and parse one chunk of designs."""
    tag = f"c{designs[0][0]:04d}"
    remote = f"{REMOTE_BASE}/{tag}"
    local = Path(tempfile.mkdtemp(prefix=f"sp_{tag}_"))
    try:
        for idx, x in designs:
            d = local / f"p{idx:04d}"
            d.mkdir()
            (d / "pa_sp.scs").write_text(sp_netlist(x))
        subprocess.run(["ssh", "-o", "BatchMode=yes", "workstation",
                        f"rm -rf {remote} && mkdir -p {remote}"],
                       check=True, timeout=60)
        subprocess.run(["scp", "-q", "-r", *[str(p) for p in local.iterdir()],
                        f"workstation:{remote}/"], check=True, timeout=120)
        run_cmd = (f"cd {remote} && for d in p*; do (cd $d && "
                   f"{SPECTRE_BIN} pa_sp.scs +log spectre.log -format psfascii "
                   f"-raw ./raw >/dev/null 2>&1); done && "
                   f"tar czf chunk.tgz p*/raw 2>/dev/null; echo TARRED")
        with _sim_lock():
            subprocess.run(["ssh", "-o", "BatchMode=yes", "workstation", run_cmd],
                           check=True, timeout=1200)
        subprocess.run(["scp", "-q", f"workstation:{remote}/chunk.tgz",
                        str(local / "chunk.tgz")], check=True, timeout=120)
        subprocess.run(["tar", "-xzf", "chunk.tgz"], cwd=local, check=True)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "workstation",
                        f"rm -rf {remote}"], timeout=60)

        rows = []
        for idx, x in designs:
            raw = local / f"p{idx:04d}" / "raw"
            row = {"idx": idx, **{c: v for c, v in zip(COLS[1:8], x)}}
            res = parse_sp(str(raw)) if raw.exists() else {"error": "no raw"}
            if "error" in res:
                row.update(S11_dB=np.nan, S22_dB=np.nan, K=np.nan, B1=np.nan,
                           error=res["error"])
            else:
                row.update(S11_dB=res["S11_dB"], S22_dB=res["S22_dB"],
                           K=res["K"], B1=res["B1"], error="")
            rows.append(row)
        return rows
    finally:
        shutil.rmtree(local, ignore_errors=True)


def main() -> None:
    designs = sobol_designs(N_TARGET)
    done: set[int] = set()
    if OUT_CSV.exists():
        done = set(pd.read_csv(OUT_CSV)["idx"].astype(int))
        print(f"[resume] {len(done)} designs already done", flush=True)

    todo = [(i, designs[i]) for i in range(N_TARGET) if i not in done]
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        rows = run_chunk(chunk)
        header = not OUT_CSV.exists()
        pd.DataFrame(rows)[COLS].to_csv(OUT_CSV, mode="a", header=header,
                                        index=False)
        n_ok = sum(1 for r in rows if not r["error"])
        print(f"[chunk] {chunk[0][0]:4d}-{chunk[-1][0]:4d}: {n_ok}/{len(rows)} ok "
              f"({len(done) + start + len(chunk)}/{N_TARGET} total)", flush=True)

    print("SP BATCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()

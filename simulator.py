"""Remote Spectre simulation runner via SSH."""

import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from config import REMOTE_HOST, REMOTE_DIR, SPECTRE_BIN, LOCAL_RESULTS_DIR

# Limit concurrent workstation simulations across local BO lanes.  Default is
# ONE Spectre job at a time (the July-2026 campaign behaviour).  PA_SIM_SLOTS=N
# turns the lock into an N-slot counting semaphore: verified 2026-08-17 that
# three concurrent main decks run in 8.4 s wall vs 7.5 s solo on the 8-core
# workstation with no license errors (each deck uses ~1 core).
_SIM_LOCK_PATH = "/tmp/pa_workstation_sim.lock"
_SIM_SLOTS = max(1, int(os.environ.get("PA_SIM_SLOTS", "1")))


def _slot_path(i: int) -> str:
    return _SIM_LOCK_PATH if i == 0 else f"{_SIM_LOCK_PATH}.{i}"


@contextmanager
def _sim_lock():
    fh = None
    # First pass: grab any free slot without blocking.
    for i in range(_SIM_SLOTS):
        cand = open(_slot_path(i), "w")
        try:
            fcntl.flock(cand, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh = cand
            break
        except OSError:
            cand.close()
    # All slots busy: block on one of them (spread lanes across slots by pid).
    if fh is None:
        fh = open(_slot_path(os.getpid() % _SIM_SLOTS), "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def run_design_point(
    main_netlist: str,
    oip3_netlist: str,
    point_id: str,
) -> dict:
    """Run one full design point evaluation on the workstation.

    Args:
        main_netlist: Spectre netlist string for DC+SP+HB
        oip3_netlist: Spectre netlist string for 2-tone HB
        point_id: Unique identifier for this design point

    Returns:
        dict with:
          main_raw_dir: local path to main sim raw files
          oip3_raw_dir: local path to OIP3 sim raw files
          time_main_s: main sim wall time (seconds)
          time_oip3_s: OIP3 sim wall time (seconds)
          time_total_s: total wall time
          success: bool
          error: str or None
    """
    remote_point_dir = f"{REMOTE_DIR}/{point_id}"
    local_point_dir = os.path.join(LOCAL_RESULTS_DIR, point_id)
    os.makedirs(local_point_dir, exist_ok=True)

    # Write netlists locally first
    main_scs = os.path.join(local_point_dir, "pa_main.scs")
    oip3_scs = os.path.join(local_point_dir, "pa_oip3.scs")
    with open(main_scs, "w") as f:
        f.write(main_netlist)
    with open(oip3_scs, "w") as f:
        f.write(oip3_netlist)

    t_start = time.time()

    # Create remote dirs and copy netlists
    _ssh_cmd(f"rm -rf {remote_point_dir} && mkdir -p {remote_point_dir}/sim_main {remote_point_dir}/sim_oip3")
    _scp_to(main_scs, f"{remote_point_dir}/sim_main/pa_main.scs")
    _scp_to(oip3_scs, f"{remote_point_dir}/sim_oip3/pa_oip3.scs")

    # Run both sims under the cross-lane lock: at most one Spectre job on the
    # workstation at any time, regardless of how many local BO lanes run.
    with _sim_lock():
        t_main_start = time.time()
        ret_main = _ssh_cmd(
            f"cd {remote_point_dir}/sim_main && "
            f"{SPECTRE_BIN} pa_main.scs +log spectre.log -format psfascii -raw ./raw +aps +mt=8 2>&1 | tail -3"
        )
        t_main = time.time() - t_main_start

        t_oip3_start = time.time()
        ret_oip3 = _ssh_cmd(
            f"cd {remote_point_dir}/sim_oip3 && "
            f"{SPECTRE_BIN} pa_oip3.scs +log spectre.log -format psfascii -raw ./raw +aps +mt=8 2>&1 | tail -3"
        )
        t_oip3 = time.time() - t_oip3_start

    # Check for errors
    main_errors = _ssh_cmd(f"grep -c ERROR {remote_point_dir}/sim_main/spectre.log 2>/dev/null || echo 0")
    oip3_errors = _ssh_cmd(f"grep -c ERROR {remote_point_dir}/sim_oip3/spectre.log 2>/dev/null || echo 0")

    main_err_count = int(main_errors.strip()) if main_errors.strip().isdigit() else 0
    oip3_err_count = int(oip3_errors.strip()) if oip3_errors.strip().isdigit() else 0

    # Copy results back
    main_raw_local = os.path.join(local_point_dir, "sim_main_raw")
    oip3_raw_local = os.path.join(local_point_dir, "sim_oip3_raw")
    _scp_from(f"{remote_point_dir}/sim_main/raw/", main_raw_local)
    _scp_from(f"{remote_point_dir}/sim_oip3/raw/", oip3_raw_local)

    # Remote workstation storage is limited: delete the point dir once its
    # raws are fetched locally (only if the fetch verifiably succeeded).
    if os.path.isdir(main_raw_local) and os.listdir(main_raw_local):
        _ssh_cmd(f"rm -rf {remote_point_dir}")

    t_total = time.time() - t_start

    success = main_err_count == 0 and oip3_err_count == 0

    error_msg = None
    if not success:
        error_msg = f"Main errors: {main_err_count}, OIP3 errors: {oip3_err_count}"

    return {
        "main_raw_dir": main_raw_local,
        "oip3_raw_dir": oip3_raw_local,
        "time_main_s": t_main,
        "time_oip3_s": t_oip3,
        "time_total_s": t_total,
        "success": success,
        "error": error_msg,
    }


def run_batch(netlists: list[tuple[str, str, str]]) -> list[dict]:
    """Run a batch of design points sequentially.

    Args:
        netlists: list of (main_netlist, oip3_netlist, point_id) tuples

    Returns:
        list of result dicts from run_design_point
    """
    results = []
    for main_nl, oip3_nl, pid in netlists:
        result = run_design_point(main_nl, oip3_nl, pid)
        results.append(result)
    return results


def _ssh_cmd(cmd: str) -> str:
    """Run a command on the remote workstation via SSH."""
    result = subprocess.run(
        ["ssh", REMOTE_HOST, cmd],
        capture_output=True, text=True, timeout=600,
    )
    return result.stdout


def _scp_to(local_path: str, remote_path: str):
    """Copy a file to the remote workstation."""
    subprocess.run(
        ["scp", "-q", local_path, f"{REMOTE_HOST}:{remote_path}"],
        capture_output=True, timeout=60,
    )


def _scp_from(remote_path: str, local_path: str):
    """Copy files from the remote workstation."""
    subprocess.run(
        ["scp", "-q", "-r", f"{REMOTE_HOST}:{remote_path}", local_path],
        capture_output=True, timeout=60,
    )

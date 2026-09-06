"""Evaluate a design point: geometry check -> netlist -> simulate -> parse."""

import numpy as np
from config import (
    BOUNDS_LOWER, BOUNDS_UPPER, D_IN_MIN_UM, IND_W, IND_SPACING,
    check_d_in,
)
from netlist import generate_main_netlist, generate_oip3_netlist
from simulator import run_design_point
from parser import parse_all


def evaluate_point(x: np.ndarray, point_id: str) -> dict:
    """Evaluate one 7D design point.

    Args:
        x: (7,) array — [L1L2_rad, L1L2_nr,
                          L3L4_rad, L3L4_nr,
                          c_wr, r_l, m_wr]
            Inductor dims in um, continuous dims in SI meters.
        point_id: unique string ID for this evaluation

    Returns:
        dict with objectives, constraints, geometry info, sim info
    """
    # 1. Extract geometry and continuous params
    L1L2_rad = float(x[0])  # um
    L1L2_nr = float(x[1])
    L3L4_rad = float(x[2])  # um
    L3L4_nr = float(x[3])
    cap_wr = float(x[4])    # meters
    res_l = float(x[5])     # meters
    mos_wr = float(x[6])    # meters

    # Snap to PDK grid: rad to nearest 1um, nr to nearest 0.5 turns
    # spiral_std_mu_z requires integer radius and half-integer turns
    L1L2_rad = round(L1L2_rad)
    L1L2_nr = round(L1L2_nr * 2) / 2
    L3L4_rad = round(L3L4_rad)
    L3L4_nr = round(L3L4_nr * 2) / 2

    # 2. Pre-sim physical validity: d_in > D_IN_MIN_UM
    d_in_L1L2 = check_d_in(L1L2_rad, L1L2_nr)
    d_in_L3L4 = check_d_in(L3L4_rad, L3L4_nr)

    _fail_parsed = {
        "objectives": {"PAE_pct": 0.0, "Gain_dB": 0.0, "Psat_dBm": 0.0},
        "constraints": {
            "K": 0.0, "B1": -1.0, "S11_dB": 0.0, "S22_dB": 0.0,
            "OIP3_dBm": 0.0, "Pdc_mW": 999.0,
        },
    }

    if d_in_L1L2 < D_IN_MIN_UM or d_in_L3L4 < D_IN_MIN_UM:
        return {
            **_fail_parsed,
            "errors": [
                f"d_in rejection: L1L2 d_in={d_in_L1L2:.1f}um, "
                f"L3L4 d_in={d_in_L3L4:.1f}um (min={D_IN_MIN_UM}um)"
            ],
            "x": x.tolist(),
            "point_id": point_id,
            "L1L2": {"rad": L1L2_rad, "nr": L1L2_nr, "d_in": d_in_L1L2},
            "L3L4": {"rad": L3L4_rad, "nr": L3L4_nr, "d_in": d_in_L3L4},
            "cap_wr_um": cap_wr * 1e6,
            "res_l_um": res_l * 1e6,
            "mos_wr_um": mos_wr * 1e6,
            "sim_time_s": 0.0,
            "rejected": True,
        }

    # 3. Generate netlists (pass geometry directly)
    main_nl = generate_main_netlist(
        L1L2_rad, L1L2_nr, L3L4_rad, L3L4_nr, cap_wr, res_l, mos_wr,
    )
    oip3_nl = generate_oip3_netlist(
        L1L2_rad, L1L2_nr, L3L4_rad, L3L4_nr, cap_wr, res_l, mos_wr,
    )

    # 4. Run simulation
    sim_result = run_design_point(main_nl, oip3_nl, point_id)

    # 5. Parse results
    if sim_result["success"]:
        try:
            parsed = parse_all(sim_result["main_raw_dir"], sim_result["oip3_raw_dir"])
        except Exception as e:
            parsed = {**_fail_parsed, "errors": [f"parse_error: {e}"]}
    else:
        parsed = {**_fail_parsed, "errors": [sim_result["error"]]}

    return {
        **parsed,
        "x": x.tolist(),
        "point_id": point_id,
        "L1L2": {"rad": L1L2_rad, "nr": L1L2_nr, "d_in": d_in_L1L2},
        "L3L4": {"rad": L3L4_rad, "nr": L3L4_nr, "d_in": d_in_L3L4},
        "cap_wr_um": cap_wr * 1e6,
        "res_l_um": res_l * 1e6,
        "mos_wr_um": mos_wr * 1e6,
        "sim_time_s": sim_result["time_total_s"],
    }


def evaluate_batch(X: np.ndarray, batch_id: int, pid_prefix: str = "") -> list[dict]:
    """Evaluate a batch of design points.

    Args:
        X: (q, 7) array of design points
        batch_id: batch number for naming
        pid_prefix: prefix for point_id

    Returns:
        list of result dicts
    """
    results = []
    for i in range(X.shape[0]):
        pfx = f"{pid_prefix}_" if pid_prefix else ""
        point_id = f"{pfx}batch{batch_id:03d}_pt{i}"
        result = evaluate_point(X[i], point_id)
        results.append(result)
    return results

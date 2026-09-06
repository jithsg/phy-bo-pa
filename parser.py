"""Parse Spectre psfascii output files to extract objectives and constraints."""

import os
import re
import math
import numpy as np
from config import F0, VDD, VBIAS, Z0


def _parse_psfascii_values(filepath: str) -> dict:
    """Parse a psfascii file and return {signal_name: value_or_list_of_values}.

    For DC files: values are real floats.
    For AC/SP/HB files: values are complex tuples (real, imag) per freq point.
    """
    if not os.path.exists(filepath):
        return {}

    with open(filepath, "r") as f:
        content = f.read()

    lines = content.split("\n")
    results = {}
    in_value = False
    current_freq = None

    for line in lines:
        s = line.strip()
        if s == "VALUE":
            in_value = True
            continue
        if s == "END":
            break
        if not in_value:
            continue

        # Frequency line
        m_freq = re.match(r'"freq"\s+([\d.eE+-]+)', s)
        if m_freq:
            current_freq = float(m_freq.group(1))
            continue

        # Complex value: "name" (real imag)
        m_cx = re.match(r'"([^"]+)"\s+\(([\d.eE+-]+)\s+([\d.eE+-]+)\)', s)
        if m_cx:
            name = m_cx.group(1)
            val = complex(float(m_cx.group(2)), float(m_cx.group(3)))
            if name not in results:
                results[name] = []
            results[name].append({"freq": current_freq, "val": val})
            continue

        # Real value with type: "name" "type" value
        m_real = re.match(r'"([^"]+)"\s+"[^"]*"\s+([\d.eE+-]+)', s)
        if m_real:
            results[m_real.group(1)] = float(m_real.group(2))
            continue

        # Real value inline (DC): "name" "type" value PROP(...)
        m_dc = re.match(r'"([^"]+)"\s+"[^"]*"\s+([\d.eE+-]+)\s+PROP', s)
        if m_dc:
            results[m_dc.group(1)] = float(m_dc.group(2))
            continue

    return results


def _find_nearest_freq(entries: list, target_freq: float) -> dict:
    """Find the entry closest to target_freq."""
    best = min(entries, key=lambda e: abs(e["freq"] - target_freq))
    return best


# ==============================================================
#  S-parameter parsing
# ==============================================================

def parse_sp(raw_dir: str) -> dict:
    """Parse S-parameter file and extract S11, S22, S21, S12 at f0.

    Also computes stability factors K and B1.

    Returns:
        dict with keys: S11_dB, S22_dB, S21_dB, S12_dB, K, B1
    """
    sp_file = os.path.join(raw_dir, "sp.sp")
    data = _parse_psfascii_values(sp_file)

    if not data or "s11" not in data:
        return {"error": "SP file not found or empty"}

    # Get S-params at f0
    s11_entry = _find_nearest_freq(data["s11"], F0)
    s21_entry = _find_nearest_freq(data["s21"], F0)
    s12_entry = _find_nearest_freq(data["s12"], F0)
    s22_entry = _find_nearest_freq(data["s22"], F0)

    s11 = s11_entry["val"]
    s21 = s21_entry["val"]
    s12 = s12_entry["val"]
    s22 = s22_entry["val"]

    s11_mag = abs(s11)
    s21_mag = abs(s21)
    s12_mag = abs(s12)
    s22_mag = abs(s22)

    # dB values
    s11_db = 20 * math.log10(s11_mag) if s11_mag > 0 else -100
    s22_db = 20 * math.log10(s22_mag) if s22_mag > 0 else -100
    s21_db = 20 * math.log10(s21_mag) if s21_mag > 0 else -100
    s12_db = 20 * math.log10(s12_mag) if s12_mag > 0 else -100

    # Rollett stability factor K
    delta = s11 * s22 - s12 * s21
    delta_mag = abs(delta)

    denom = 2 * s12_mag * s21_mag
    if denom > 1e-30:
        K = (1 - s11_mag**2 - s22_mag**2 + delta_mag**2) / denom
    else:
        K = 999.0  # Very stable (no reverse gain)

    # B1 stability factor
    B1 = 1 + s11_mag**2 - s22_mag**2 - delta_mag**2

    return {
        "S11_dB": s11_db,
        "S22_dB": s22_db,
        "S21_dB": s21_db,
        "S12_dB": s12_db,
        "K": K,
        "B1": B1,
    }


# ==============================================================
#  DC operating point parsing
# ==============================================================

def parse_dc(raw_dir: str) -> dict:
    """Parse DC operating point.

    Returns:
        dict with keys: Vds_M0, Vgs_M0, Ids_M0, Pdc_mW, Vdd_current_A
    """
    dc_file = os.path.join(raw_dir, "dcOp.dc")
    data = _parse_psfascii_values(dc_file)

    if not data:
        return {"error": "DC file not found or empty"}

    # MOSFET operating point
    vds = data.get("M0.di", VDD) - data.get("M0.si", 0.0)
    vgs = data.get("M0.gi", VBIAS) - data.get("M0.si", 0.0)

    # DC current from VDD supply
    i_vdd = abs(data.get("V1:p", 0.0))
    pdc_mw = VDD * i_vdd * 1000  # mW

    return {
        "Vds_M0": vds,
        "Vgs_M0": vgs,
        "Ids_M0": data.get("M0:1", 0.0),
        "Pdc_mW": pdc_mw,
        "Idd_A": i_vdd,
    }


# ==============================================================
#  HB sweep parsing (PAE, Gain, Psat)
# ==============================================================

def parse_hb_sweep(raw_dir: str) -> dict:
    """Parse HB sweep results to extract PAE, Gain, Psat.

    The sweep goes from Pin=-40 to +10 dBm (200 points).
    Each sweep point has: sweephb-NNN_hb.fd.pss_hb

    Returns:
        dict with keys: PAE_pct, Gain_dB, Psat_dBm, Pin_at_Psat_dBm
        Also: pin_array, pout_array, gain_array, pae_array for plotting
    """
    # Find all HB sweep files
    sweep_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.startswith("sweephb-") and f.endswith("_hb.fd.pss_hb")
    ])

    if not sweep_files:
        return {"error": "No HB sweep files found"}

    n_points = len(sweep_files)
    # Pin sweep: -40 to 10 dBm, 200 points (lin)
    pin_dbm = np.linspace(-40, 10, n_points)

    pout_dbm = np.full(n_points, np.nan)
    pdc_w = np.full(n_points, np.nan)

    for i, fname in enumerate(sweep_files):
        filepath = os.path.join(raw_dir, fname)
        data = _parse_psfascii_values(filepath)

        if not data:
            continue

        # PORT1:p is the output port current (at fundamental)
        # PORT0:p is the input port current
        # V1:p is VDD supply current

        # Find fundamental frequency component
        if "PORT1:p" in data and isinstance(data["PORT1:p"], list):
            # Find the entry at f0 (fundamental)
            p1_entry = _find_nearest_freq(data["PORT1:p"], F0)
            i_out = p1_entry["val"]
            # Output power: Pout = |I|^2 * R / 2 (peak current to avg power)
            # In HB, port current is already the complex amplitude
            pout_w = 0.5 * abs(i_out)**2 * Z0
            if pout_w > 0:
                pout_dbm[i] = 10 * math.log10(pout_w * 1000)  # dBm
        elif "PORT1:p" in data:
            # Single complex value
            i_out = data["PORT1:p"]
            if isinstance(i_out, complex):
                pout_w = 0.5 * abs(i_out)**2 * Z0
                if pout_w > 0:
                    pout_dbm[i] = 10 * math.log10(pout_w * 1000)

        # DC power from V1:p
        if "V1:p" in data:
            if isinstance(data["V1:p"], list):
                # DC component (freq=0)
                dc_entries = [e for e in data["V1:p"] if e["freq"] == 0 or e["freq"] < 1e6]
                if dc_entries:
                    i_dc = abs(dc_entries[0]["val"].real)
                    pdc_w[i] = VDD * i_dc
            elif isinstance(data["V1:p"], (int, float)):
                pdc_w[i] = VDD * abs(data["V1:p"])

    # Compute Gain and PAE
    valid = ~np.isnan(pout_dbm)
    if not np.any(valid):
        return {"error": "No valid HB data points"}

    pin_w = 10 ** ((pin_dbm - 30) / 10)  # Convert dBm to W
    pout_w_arr = 10 ** ((pout_dbm - 30) / 10)

    gain_db = pout_dbm - pin_dbm

    # PAE = (Pout - Pin) / Pdc * 100
    pae = np.full(n_points, np.nan)
    for i in range(n_points):
        if not np.isnan(pout_dbm[i]) and not np.isnan(pdc_w[i]) and pdc_w[i] > 0:
            pae[i] = (pout_w_arr[i] - pin_w[i]) / pdc_w[i] * 100

    # Find Psat: where gain compresses by 1 dB from small-signal
    # Small-signal gain = gain at lowest Pin
    valid_gain = ~np.isnan(gain_db)
    if np.any(valid_gain):
        ss_gain = np.nanmean(gain_db[valid_gain][:10])  # Average of first 10 points

        # Find 1-dB compression point
        p1db_idx = None
        for i in range(n_points):
            if valid_gain[i] and gain_db[i] < ss_gain - 1.0:
                p1db_idx = i
                break

        # Psat: maximum output power
        psat_idx = np.nanargmax(pout_dbm)
        psat_dbm = pout_dbm[psat_idx]

        # Max PAE
        pae_max_idx = np.nanargmax(pae)
        pae_max = pae[pae_max_idx]

        # Gain at max PAE point (or small-signal gain)
        gain_at_pae_max = gain_db[pae_max_idx] if not np.isnan(gain_db[pae_max_idx]) else ss_gain
    else:
        ss_gain = np.nan
        psat_dbm = np.nan
        pae_max = np.nan
        gain_at_pae_max = np.nan

    return {
        "PAE_pct": float(pae_max) if not np.isnan(pae_max) else 0.0,
        "Gain_dB": float(ss_gain) if not np.isnan(ss_gain) else 0.0,
        "Psat_dBm": float(psat_dbm) if not np.isnan(psat_dbm) else 0.0,
        "pin_dbm": pin_dbm,
        "pout_dbm": pout_dbm,
        "gain_db": gain_db,
        "pae_pct": pae,
    }


# ==============================================================
#  OIP3 parsing (2-tone HB)
# ==============================================================

def parse_oip3(raw_dir: str) -> dict:
    """Parse 2-tone HB results to extract OIP3.

    OIP3 = Pout_fund + (Pout_fund - Pout_IM3) / 2

    Returns:
        dict with keys: OIP3_dBm, Pout_fund_dBm, Pout_IM3_dBm
    """
    hb_file = os.path.join(raw_dir, "hb.fd.qpss_hb")
    data = _parse_psfascii_values(hb_file)

    if not data or "PORT1:p" not in data:
        return {"error": "OIP3 HB file not found or empty"}

    port1_entries = data["PORT1:p"]
    if not isinstance(port1_entries, list):
        return {"error": "PORT1:p data format unexpected"}

    f1 = F0
    f2 = F0 + 1e6  # f1 + 1 MHz
    f_im3_low = 2 * f1 - f2   # 2*f1 - f2
    f_im3_high = 2 * f2 - f1  # 2*f2 - f1

    # Extract port current at each frequency
    def get_power_at_freq(entries, target_freq):
        entry = _find_nearest_freq(entries, target_freq)
        if abs(entry["freq"] - target_freq) > 0.5e6:  # Tolerance 0.5 MHz
            return None
        i_out = entry["val"]
        p_w = 0.5 * abs(i_out)**2 * Z0
        if p_w > 0:
            return 10 * math.log10(p_w * 1000)  # dBm
        return None

    p_f1 = get_power_at_freq(port1_entries, f1)
    p_f2 = get_power_at_freq(port1_entries, f2)
    p_im3_low = get_power_at_freq(port1_entries, f_im3_low)
    p_im3_high = get_power_at_freq(port1_entries, f_im3_high)

    if p_f1 is None:
        return {"error": "Could not extract fundamental power"}

    # Use average of f1 and f2 fundamentals
    p_fund = p_f1
    if p_f2 is not None:
        p_fund = (p_f1 + p_f2) / 2

    # Use the stronger IM3 product
    p_im3 = None
    if p_im3_low is not None and p_im3_high is not None:
        p_im3 = max(p_im3_low, p_im3_high)
    elif p_im3_low is not None:
        p_im3 = p_im3_low
    elif p_im3_high is not None:
        p_im3 = p_im3_high

    if p_im3 is None:
        return {"error": "Could not extract IM3 power", "Pout_fund_dBm": p_fund}

    # OIP3 = Pout_fund + (Pout_fund - Pout_IM3) / 2
    oip3 = p_fund + (p_fund - p_im3) / 2

    return {
        "OIP3_dBm": oip3,
        "Pout_fund_dBm": p_fund,
        "Pout_IM3_dBm": p_im3,
    }


# ==============================================================
#  Combined parser
# ==============================================================

def parse_all(main_raw_dir: str, oip3_raw_dir: str) -> dict:
    """Parse all simulation results and return objectives + constraints.

    Returns:
        dict with:
          objectives: {PAE_pct, Gain_dB, Psat_dBm}
          constraints: {K, B1, S11_dB, S22_dB, OIP3_dBm, Pdc_mW, Vds_M0}
          raw: all parsed data
          feasible: bool
          error: str or None
    """
    sp_data = parse_sp(main_raw_dir)
    dc_data = parse_dc(main_raw_dir)
    hb_data = parse_hb_sweep(main_raw_dir)
    oip3_data = parse_oip3(oip3_raw_dir)

    errors = []
    for name, d in [("SP", sp_data), ("DC", dc_data), ("HB", hb_data), ("OIP3", oip3_data)]:
        if "error" in d:
            errors.append(f"{name}: {d['error']}")

    objectives = {
        "PAE_pct": hb_data.get("PAE_pct", 0.0),
        "Gain_dB": hb_data.get("Gain_dB", 0.0),
        "Psat_dBm": hb_data.get("Psat_dBm", 0.0),
    }

    constraints = {
        "K": sp_data.get("K", 0.0),
        "B1": sp_data.get("B1", -1.0),
        "S11_dB": sp_data.get("S11_dB", 0.0),
        "S22_dB": sp_data.get("S22_dB", 0.0),
        "OIP3_dBm": oip3_data.get("OIP3_dBm", 0.0),
        "Pdc_mW": dc_data.get("Pdc_mW", 999.0),
        "Vds_M0": dc_data.get("Vds_M0", 0.0),
    }

    return {
        "objectives": objectives,
        "constraints": constraints,
        "raw": {"sp": sp_data, "dc": dc_data, "hb": hb_data, "oip3": oip3_data},
        "errors": errors if errors else None,
    }

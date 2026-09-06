"""Generate Spectre netlists for PA simulation.

Direct geometry: takes (rad, nr) per inductor role directly.
No inductor table lookup — Spectre evaluates the exact geometry.
"""

from config import (
    PDK_INCLUDE, BALUN_VA, VDD, VBIAS, F0,
    CAP_BR, CAP_GR, RES_W, RES_MULTI,
    MOS_LR, MOS_NR, MOS_MULTI,
    IND_W, IND_SPACING, IND_LAY, IND_GDIS,
)


def _fmt_um(val_m: float) -> str:
    """Format a value in meters as um string for netlist."""
    return f"{val_m * 1e6:.4f}u"


def generate_main_netlist(
    L1L2_rad: float,
    L1L2_nr: float,
    L3L4_rad: float,
    L3L4_nr: float,
    cap_wr_m: float,
    res_l_m: float,
    mos_wr_m: float,
) -> str:
    """Generate main sim netlist (DC + SP + HB sweep).

    Args:
        L1L2_rad: gate inductor radius (um)
        L1L2_nr: gate inductor number of turns
        L3L4_rad: drain inductor radius (um)
        L3L4_nr: drain inductor number of turns
        cap_wr_m: moscap_rf wr=lr in meters
        res_l_m: rppolywo l in meters
        mos_wr_m: nmos_rf wr in meters

    Returns:
        Netlist string
    """
    cap_wr = _fmt_um(cap_wr_m)
    res_l = _fmt_um(res_l_m)
    res_w = _fmt_um(RES_W)
    mos_wr = _fmt_um(mos_wr_m)
    mos_lr = f"{MOS_LR * 1e9:.0f}n"

    f0_g = F0 * 1e-9

    return f"""\
simulator lang=spectre
global 0
parameters f1={f0_g}G pin=-10 vbias={VBIAS}

include "{PDK_INCLUDE}" section=top_tt

// Push-pull PA core
M0 (RFout_n inp vss vss) nmos_rf lr={mos_lr} wr={mos_wr} nr={MOS_NR} multi={MOS_MULTI}
M1 (RFout_p inn vss vss) nmos_rf lr={mos_lr} wr={mos_wr} nr={MOS_NR} multi={MOS_MULTI}

R4 (mid_inp inp) rppolywo l={res_l} w={res_w} m=1 multi={RES_MULTI}
R2 (mid_inn inn) rppolywo l={res_l} w={res_w} m=1 multi={RES_MULTI}

L32 (RFout_n vdd vss) spiral_std_mu_z w={IND_W}u rad={L3L4_rad}u nr={L3L4_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u
L29 (vdd RFout_p vss) spiral_std_mu_z w={IND_W}u rad={L3L4_rad}u nr={L3L4_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u
L22 (vbias_node mid_inp vss) spiral_std_mu_z w={IND_W}u rad={L1L2_rad}u nr={L1L2_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u
L21 (vbias_node mid_inn vss) spiral_std_mu_z w={IND_W}u rad={L1L2_rad}u nr={L1L2_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u

C1 (mid_inp RFip vss) moscap_rf wr={cap_wr} lr={cap_wr} br={CAP_BR} gr={CAP_GR} multi=1
C0 (mid_inn RFin vss) moscap_rf wr={cap_wr} lr={cap_wr} br={CAP_BR} gr={CAP_GR} multi=1

// Baluns
I28 (_net0 RFout_n vdd RFout_p) balun_ideal n21=0.707
I18 (in RFip vss RFin) balun_ideal n21=1.414

// Ports and sources
PORT1 (_net0 0) port r=50 num=2 type=dc freq=f1 dbm=pin
PORT0 (in 0) port r=50 num=1 type=sine freq=f1 dbm=pin
V2 (vbias_node 0) vsource dc=vbias type=dc
V1 (vdd 0) vsource dc={VDD} type=dc
V0 (vss 0) vsource dc=0 type=dc

// Analyses
dcOp dc write="spectre.dc" maxiters=150 maxsteps=10000 annotate=status
dcOpInfo info what=oppoint where=rawfile
sp sp ports=[PORT0 PORT1] start=0.1G stop=5G lin=200 annotate=status
sweephb sweep param=pin start=-40 stop=10 lin=200 {{
  hb hb autoharms=yes autotstab=yes oversample=[1] \\
     fundfreqs=[(f1)] maxharms=[5] errpreset=moderate annotate=status
}}

simulatorOptions options psfversion="1.4.0" reltol=1e-3 vabstol=1e-6 \\
    iabstol=1e-12 temp=27 tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 \\
    maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 checklimitdest=psf

save M0:1 M1:1 in RFip RFin mid_inp mid_inn inp inn RFout_n RFout_p _net0 \\
    PORT1:p PORT0:p V1:p
saveOptions options save=allpub
ahdl_include "{BALUN_VA}"
"""


def generate_oip3_netlist(
    L1L2_rad: float,
    L1L2_nr: float,
    L3L4_rad: float,
    L3L4_nr: float,
    cap_wr_m: float,
    res_l_m: float,
    mos_wr_m: float,
) -> str:
    """Generate 2-tone HB netlist for OIP3."""
    cap_wr = _fmt_um(cap_wr_m)
    res_l = _fmt_um(res_l_m)
    res_w = _fmt_um(RES_W)
    mos_wr = _fmt_um(mos_wr_m)
    mos_lr = f"{MOS_LR * 1e9:.0f}n"

    f0_g = F0 * 1e-9
    f2_g = f0_g + 0.001  # f2 = f1 + 1 MHz

    return f"""\
simulator lang=spectre
global 0
parameters f1={f0_g}G f2={f2_g}G pin=-20 vbias={VBIAS}

include "{PDK_INCLUDE}" section=top_tt

// Push-pull PA core
M0 (RFout_n inp vss vss) nmos_rf lr={mos_lr} wr={mos_wr} nr={MOS_NR} multi={MOS_MULTI}
M1 (RFout_p inn vss vss) nmos_rf lr={mos_lr} wr={mos_wr} nr={MOS_NR} multi={MOS_MULTI}

R4 (mid_inp inp) rppolywo l={res_l} w={res_w} m=1 multi={RES_MULTI}
R2 (mid_inn inn) rppolywo l={res_l} w={res_w} m=1 multi={RES_MULTI}

L32 (RFout_n vdd vss) spiral_std_mu_z w={IND_W}u rad={L3L4_rad}u nr={L3L4_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u
L29 (vdd RFout_p vss) spiral_std_mu_z w={IND_W}u rad={L3L4_rad}u nr={L3L4_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u
L22 (vbias_node mid_inp vss) spiral_std_mu_z w={IND_W}u rad={L1L2_rad}u nr={L1L2_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u
L21 (vbias_node mid_inn vss) spiral_std_mu_z w={IND_W}u rad={L1L2_rad}u nr={L1L2_nr} \
    lay={IND_LAY} gdis={IND_GDIS}u multi=1 spacing={IND_SPACING}u

C1 (mid_inp RFip vss) moscap_rf wr={cap_wr} lr={cap_wr} br={CAP_BR} gr={CAP_GR} multi=1
C0 (mid_inn RFin vss) moscap_rf wr={cap_wr} lr={cap_wr} br={CAP_BR} gr={CAP_GR} multi=1

// Baluns
I28 (_net0 RFout_n vdd RFout_p) balun_ideal n21=0.707
I18 (in RFip vss RFin) balun_ideal n21=1.414

// Ports - 2 tone
PORT1 (_net0 0) port r=50 num=2 type=dc freq=f1 dbm=pin
PORT0 (in 0) port r=50 num=1 type=sine freq=f1 dbm=pin freq2=f2 dbm2=pin
V2 (vbias_node 0) vsource dc=vbias type=dc
V1 (vdd 0) vsource dc={VDD} type=dc
V0 (vss 0) vsource dc=0 type=dc

// 2-tone HB for OIP3
hb hb autoharms=yes autotstab=yes oversample=[1] \\
   fundfreqs=[(f1) (f2)] maxharms=[5 5] errpreset=moderate annotate=status

simulatorOptions options psfversion="1.4.0" reltol=1e-3 vabstol=1e-6 \\
    iabstol=1e-12 temp=27 tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 \\
    maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 checklimitdest=psf

save M0:1 M1:1 PORT1:p PORT0:p V1:p
saveOptions options save=allpub
ahdl_include "{BALUN_VA}"
"""

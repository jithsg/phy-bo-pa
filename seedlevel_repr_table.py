"""Reproduce table (c) HVs and compute the seed-level representation table."""
import json, sys
from pathlib import Path
import numpy as np, torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
R = Path("results")
OBJ = ["PAE_pct","Gain_dB","Psat_dBm"]
def load(d): return json.load(open(R/d/"all_results.json"))
def objs(recs): return np.array([[r["objectives"][k] for k in OBJ] for r in recs])
def feas(recs): return np.array([bool(r.get("feasible")) for r in recs])
def hv(Y, ref):
    if len(Y)==0: return 0.0
    part = FastNondominatedPartitioning(ref_point=torch.tensor(ref,dtype=torch.double), Y=torch.tensor(Y,dtype=torch.double))
    return part.compute_hypervolume().item()
def refpt(Y): 
    lo, hi = Y.min(0), Y.max(0); return lo - 0.1*np.maximum(hi-lo,1e-6)
def bo_phase(recs): return [r for r in recs if "doe" not in r["point_id"]]
def summarize(name, arm_recs, ctrl_recs, n=100):
    a, c = arm_recs[:n], ctrl_recs[:n]
    Ya, Yc = objs(a), objs(c)
    fa, fc = feas(a), feas(c)
    out = {}
    for tag, refY in (("ref=union all", np.vstack([Ya,Yc])), ("ref=union feasible", np.vstack([Ya[fa],Yc[fc]]) if (fa.any() or fc.any()) else np.vstack([Ya,Yc]))):
        ref = refpt(refY)
        out[tag] = (hv(Ya[fa],ref), hv(Yc[fc],ref))
    fa_bo = sum(1 for r in bo_phase(a) if r.get("feasible")); fc_bo = sum(1 for r in bo_phase(c) if r.get("feasible"))
    bpa = Ya[fa][:,0].max() if fa.any() else float("nan"); bpc = Yc[fc][:,0].max() if fc.any() else float("nan")
    return dict(feas_bo=(fa_bo,fc_bo), feas_tot=(int(fa.sum()),int(fc.sum())), bestPAE=(bpa,bpc), hv=out, n=(len(a),len(c)))
ctrl = {42:"geo2_warm_s42", 7:"ms_stage1_s7", 11:"ms_stage1_s11", 23:"ms_stage1_s23", 101:"ms_stage1_s101"}
print("=== (c) primed vs control — should reproduce 35527/39778, 30023/33742, 27582/25122, 20950/29558, 25161/27940 ===")
for s in (42,7,11,23,101):
    r = summarize(s, load(f"primed_cold_s{s}"), load(ctrl[s]))
    print(s, "feas_bo", r["feas_bo"], "bestPAE %.1f/%.1f"%r["bestPAE"], {k:("%.0f/%.0f"%v) for k,v in r["hv"].items()}, "n", r["n"])
print("\n=== representation: physfeat vs control ===")
rows=[]
for s in (42,7,11,23,101):
    r = summarize(s, load(f"physfeat_cold_s{s}"), load(ctrl[s]))
    print(s, "feas_bo", r["feas_bo"], "feas_tot", r["feas_tot"], "bestPAE %.1f/%.1f"%r["bestPAE"], {k:("%.0f/%.0f"%v) for k,v in r["hv"].items()}, "n", r["n"])
    rows.append((s,r))
json.dump([(s,{k:(v if k!="hv" else {kk:list(vv) for kk,vv in v.items()}) for k,v in r.items()}) for s,r in rows], open(sys.argv[1] if len(sys.argv)>1 else "/dev/null","w"), indent=1, default=float)

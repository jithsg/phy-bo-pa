"""Validation of the πBO-style prior-weighted qLogNEHVI before any Spectre run.

Runs on the seed-42 DoE (20 already simulated points of the zero-mean reference)
and needs no simulator.  Checks:
  1. uniform π reproduces the plain qLogNEHVI values and ranking;
  2. log π is largest at the manual design;
  3. the prior influence (γ_n = β/n and the acquisition shift) decreases with n;
  4. gradients of the weighted acquisition are finite;
  5. candidate generation, PDK snapping and d_in validity still work;
  6. the prior is built from config constants only (no result-derived input).

Usage: python3 pibo_validate.py [seed]   (writes results/pibo_validation_s{seed}.txt)
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from scipy.stats import spearmanr

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import bo_loop_geo as G  # noqa: E402
from config import BOUNDS_LOWER, BOUNDS_UPPER, LOCAL_RESULTS_DIR, N_DIMS, PRESIM_X, OptimizationConfig  # noqa: E402
from pibo_prior import PriorSpec, PriorWeightedAcqf  # noqa: E402

torch.manual_seed(0)
DT = G.DTYPE
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def fit_on_doe(seed: int):
    src = Path(LOCAL_RESULTS_DIR) / "kbs_results" / f"ext_s{seed}" / f"sw_stage1ext_s{seed}" / "all_results.json"
    pts = json.load(open(src))[:20]
    valid = [r for r in pts if not r.get("rejected")]
    X = np.array([r["x"] for r in valid])
    Yo = np.array([[r["objectives"][k] for k in G.OBJ_NAMES] for r in valid])
    Yc_raw = np.array([[r["constraints"][n] for n, _, _ in G.CONSTRAINT_TRANSFORMS] for r in valid])
    Yc = np.empty_like(Yc_raw)
    for ci, (_, sc, off) in enumerate(G.CONSTRAINT_TRANSFORMS):
        Yc[:, ci] = Yc_raw[:, ci] * sc + off
    train_X = normalize(torch.tensor(X, dtype=DT), G.BOUNDS)
    train_Y = torch.tensor(np.hstack([Yo, Yc]), dtype=DT)
    model, _ = G.build_model_list(train_X, train_Y, use_physics=False, use_physics_features=False)
    ref = []
    for j in range(3):
        col = Yo[:, j]
        ref.append(col.min() - 0.1 * max(col.max() - col.min(), 1e-6))
    return model, train_X, torch.tensor(ref, dtype=DT), src


def base_acqf(model, train_X, ref):
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.sampling.normal import SobolQMCNormalSampler
    cons = [lambda Y, idx=idx: Y[..., idx] for idx in range(3, 3 + G.N_CONSTRAINTS)]
    return qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=ref.tolist(), X_baseline=train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
        constraints=cons, prune_baseline=True, cache_root=False)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    model, train_X, ref, src = fit_on_doe(seed)
    prior = PriorSpec.manual_design(n_iterations=160)
    unif = PriorSpec.uniform(n_iterations=160)
    say(f"piBO validation on the seed-{seed} DoE ({src})")
    say(f"prior spec: {json.dumps(prior.as_dict())}")
    ok_all = True

    # ---- 1. uniform prior == plain qLogNEHVI
    acq = base_acqf(model, train_X, ref)
    Xr = torch.rand(512, 1, N_DIMS, dtype=DT)
    with torch.no_grad():
        a0 = acq(Xr)
        a_unif = PriorWeightedAcqf(acq, unif, n=1)(Xr)
        a_prior_late = PriorWeightedAcqf(acq, prior, n=10 ** 9)(Xr)
    d1 = (a0 - a_unif).abs().max().item()
    rho_u = spearmanr(a0.numpy(), a_unif.numpy()).correlation
    d_late = (a0 - a_prior_late).abs().max().item()
    rho_l = spearmanr(a0.numpy(), a_prior_late.numpy()).correlation
    ok1 = d1 == 0.0 and rho_u == 1.0 and d_late < 1e-6
    ok_all &= ok1
    say(f"[1] uniform prior: max|Δ| = {d1:.3e}, Spearman ρ = {rho_u:.6f}; "
        f"Gaussian prior with n→∞: max|Δ| = {d_late:.3e}, ρ = {rho_l:.6f}  -> {'PASS' if ok1 else 'FAIL'}")

    # ---- 2. prior mode at the manual design
    x0 = torch.tensor(prior.center, dtype=DT)
    Xs = torch.rand(200_000, N_DIMS, dtype=DT)
    with torch.no_grad():
        lp0 = prior.log_prior(x0.unsqueeze(0)).item()
        lps = prior.log_prior(Xs)
        lp_corners = prior.log_prior(torch.tensor([[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=DT))
    g = x0.clone().requires_grad_(True)
    prior.log_prior(g.unsqueeze(0)).sum().backward()
    ok2 = bool(lp0 >= lps.max().item() and g.grad.abs().max().item() < 1e-8 and torch.isfinite(lp_corners).all().item())
    ok_all &= ok2
    say(f"[2] log π(manual) = {lp0:.4f} ≥ max over 200k random points {lps.max().item():.4f}; "
        f"∇log π at the manual design = {g.grad.abs().max().item():.1e}; "
        f"log π at the box corners = {lp_corners.tolist()} (finite, floor log ε = {np.log(prior.eps):.1f})  -> {'PASS' if ok2 else 'FAIL'}")
    say(f"    manual design (raw) = {list(map(float, PRESIM_X))}, normalised = {[round(c, 3) for c in prior.center]}")

    # ---- 3. decreasing influence with n
    with torch.no_grad():
        lp = prior.log_prior(Xr.squeeze(1))
    shifts = []
    for n in (1, 2, 4, 8, 16, 32, 64, 160):
        w = prior.weight(n)
        shift = (w * (lp - lp.max())).abs().mean().item()   # mean penalty relative to the prior mode
        shifts.append((n, w, shift))
    mono = all(shifts[i][1] > shifts[i + 1][1] and shifts[i][2] > shifts[i + 1][2] for i in range(len(shifts) - 1))
    ok_all &= mono
    say("[3] prior exponent γ_n = β/n and mean |γ_n·(log π − log π_max)| over random points:")
    for n, w, s in shifts:
        say(f"    n={n:4d}  γ_n={w:7.4f}  mean shift={s:9.3f}")
    say(f"    strictly decreasing in n  -> {'PASS' if mono else 'FAIL'}")

    # ---- 4. finite gradients
    acqw = PriorWeightedAcqf(acq, prior, n=1)
    Xg = torch.rand(64, 1, N_DIMS, dtype=DT).requires_grad_(True)
    val = acqw(Xg)
    val.sum().backward()
    gmax = Xg.grad.abs().max().item()
    x0g = x0.view(1, 1, -1).clone().requires_grad_(True)
    acqw(x0g).sum().backward()
    ok4 = bool(torch.isfinite(val).all().item() and torch.isfinite(Xg.grad).all().item() and torch.isfinite(x0g.grad).all().item())
    ok_all &= ok4
    say(f"[4] weighted acquisition finite on 64 random points: {torch.isfinite(val).all().item()}; "
        f"gradients finite: {torch.isfinite(Xg.grad).all().item()} (max |∂| = {gmax:.3e}); "
        f"finite at the manual design: {torch.isfinite(x0g.grad).all().item()}  -> {'PASS' if ok4 else 'FAIL'}")

    # ---- 5. candidate generation + snapping + validity through the real code path
    cfg = OptimizationConfig(n_initial=20, n_iterations=160, batch_size=1, seed=seed,
                             num_restarts=5, raw_samples=256)
    bounds_norm = torch.tensor([[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=DT)
    rows = []
    ok5 = True
    for n in (1, 16, 160):
        cand = G.get_candidates(model, train_X, ref, bounds_norm, cfg,
                                use_din_constraint=True, pibo_prior=prior, pibo_n=n)
        raw = unnormalize(cand, G.BOUNDS).numpy()[0]
        snapped = G.snap_to_pdk_grid(raw)
        valid = G.check_din_valid(snapped)
        dist = float(np.linalg.norm(cand.numpy()[0] - np.array(prior.center)))
        ok5 &= bool(valid) and bool(np.all(snapped >= BOUNDS_LOWER)) and bool(np.all(snapped <= BOUNDS_UPPER))
        rows.append((n, dist, snapped, valid))
    say("[5] optimize_acqf with the d_in inequality, PDK snap, validity (one candidate each):")
    for n, dist, snapped, valid in rows:
        say(f"    n={n:3d}: ‖x_norm − manual‖₂ = {dist:.3f}; snapped = "
            f"[{snapped[0]:.0f}, {snapped[1]:.1f}, {snapped[2]:.0f}, {snapped[3]:.1f}, "
            f"{snapped[4]:.2e}, {snapped[5]:.2e}, {snapped[6]:.2e}]; d_in valid = {valid}")
    say(f"    all candidates valid and inside the bounds  -> {'PASS' if ok5 else 'FAIL'}")
    ok_all &= ok5

    # ---- 6. no result-derived information in the prior
    tree = ast.parse(open(BASE / "pibo_prior.py").read())
    imports = sorted({(n.module if isinstance(n, ast.ImportFrom) else a.name)
                      for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                      for a in (n.names if isinstance(n, ast.Import) else [None])})
    opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("open",)]
    cfg_names = {"config"}
    ok6 = ("config" in imports and not opens and "results" not in open(BASE / "pibo_prior.py").read().lower().replace("result file", ""))
    ok_all &= ok6
    say(f"[6] pibo_prior.py imports {imports}; file reads: {len(opens)}; prior centre = config.PRESIM_X "
        f"(manual design, PAE 53.3 %), σ = {prior.sigma} of each domain, ε = {prior.eps}, β = N/10 = {prior.beta}  "
        f"-> {'PASS' if ok6 else 'FAIL'}")

    say(f"\nOVERALL: {'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'}")
    out = Path(LOCAL_RESULTS_DIR) / f"pibo_validation_s{seed}.txt"
    out.write_text("\n".join(LINES) + "\n")
    say(f"written {out}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

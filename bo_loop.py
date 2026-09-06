"""Multi-Objective Bayesian Optimization for PA Direct Geometry.

7D all-continuous: [L1L2_rad, L1L2_nr, L3L4_rad, L3L4_nr, C_wr, R_l, M_wr]
3 objectives (PAE, Gain, Psat), 6 soft constraints via qLogNEHVI callables.
2 hard pre-sim checks: bounds (BoTorch) + d_in > D_IN_MIN_UM (evaluate.py; value withheld under the PDK NDA).

No inductor encoding, no snapping, no pair restriction, no dedup.
Spectre evaluates every geometry directly.
"""

import os
import json
import time
import logging
import numpy as np
import torch
import botorch.models.transforms.input
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from config import (
    BOUNDS_LOWER, BOUNDS_UPPER, CONSTRAINTS, LOCAL_RESULTS_DIR,
    DIM_NAMES, N_DIMS, PRESIM_X, PRESIM_OBJECTIVES,
    OptimizationConfig,
)
from evaluate import evaluate_batch, evaluate_point

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

BOUNDS = torch.tensor(
    np.stack([BOUNDS_LOWER, BOUNDS_UPPER]), dtype=DTYPE, device=DEVICE
)

# Objective names (all maximized)
OBJ_NAMES = ["PAE_pct", "Gain_dB", "Psat_dBm"]

# Constraint transforms: c(x) >= 0 means feasible
CONSTRAINT_TRANSFORMS = [
    ("K",        1.0,  -CONSTRAINTS["K_min"]),         # K - 1 >= 0
    ("B1",       1.0,  -CONSTRAINTS["B1_min"]),         # B1 - 0 >= 0
    ("S11_dB",  -1.0,   CONSTRAINTS["S11_max_dB"]),     # -S11 + (-8) >= 0
    ("S22_dB",  -1.0,   CONSTRAINTS["S22_max_dB"]),     # -S22 + (-8) >= 0
    ("OIP3_dBm", 1.0,  -CONSTRAINTS["OIP3_min_dBm"]),  # OIP3 - 18 >= 0
    ("Pdc_mW",  -1.0,   CONSTRAINTS["Pdc_max_mW"]),    # -Pdc + 25 >= 0
]
N_CONSTRAINTS = len(CONSTRAINT_TRANSFORMS)
N_OBJECTIVES = 3

# ---------------------------------------------------------------
# Adaptive Constraint Relaxation (ACR)
# ---------------------------------------------------------------
# Max relaxation per constraint (in transformed units, where >= 0 = feasible).
# A positive slack value means the acquisition function temporarily accepts
# designs that violate the true spec by up to this amount.
# The GP always trains on the TRUE constraint values — only the acqf
# feasibility threshold is shifted.
#
# These values are calibrated from violation rates in the DoE:
#   S11: 72% violation → 3.0 dB relaxation (hardest)
#   S22: 62% violation → 3.0 dB relaxation
#   Pdc: 28% violation → 5.0 mW relaxation
#   OIP3: 24% violation → 3.0 dBm relaxation
#   K:   13% violation → 0.15 relaxation
#   B1:   0% violation → 0.0 (no relaxation needed)

ACR_MAX_SLACK = [
    0.10,   # K:    allow K >= 0.90 initially  (true spec: K >= 1.0)
    0.0,    # B1:   no relaxation (0% violation)
    1.5,    # S11:  allow S11 <= -6.5 dB initially (true spec: <= -8 dB)
    1.5,    # S22:  allow S22 <= -6.5 dB initially (true spec: <= -8 dB)
    1.5,    # OIP3: allow OIP3 >= 16.5 dBm initially (true spec: >= 18 dBm)
    3.0,    # Pdc:  allow Pdc <= 28 mW initially (true spec: <= 25 mW)
]


def get_acr_slack(iteration: int, n_iterations: int) -> list[float]:
    """Compute per-constraint relaxation slack using exponential decay.

    Returns a list of non-negative slack values. At iteration 0, slack is at
    maximum (most relaxed). Decays exponentially so that constraints reach
    ~95% of full spec by the 1/3 mark, focusing most of the budget on
    near-spec exploration.

    Args:
        iteration: current BO iteration (0-indexed)
        n_iterations: total number of BO iterations

    Returns:
        List of slack values, one per constraint.
    """
    import math
    # progress: 0.0 at start → 1.0 at end
    progress = iteration / max(n_iterations - 1, 1)
    # Exponential decay with time constant τ = 0.2
    # At progress=0.2 (~iter 16/80): decay ≈ 0.37 (63% tightened)
    # At progress=0.4 (~iter 32/80): decay ≈ 0.14 (86% tightened)
    # At progress=0.6 (~iter 48/80): decay ≈ 0.05 (95% tightened)
    tau = 0.2
    decay = math.exp(-progress / tau)
    return [s * decay for s in ACR_MAX_SLACK]


# ---------------------------------------------------------------
# DoE generation
# ---------------------------------------------------------------

def generate_initial_doe(
    n_points: int,
    seed: int = 42,
    warmstart_frac: float = 0.5,
    warmstart_centers: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Generate initial design of experiments.

    If warmstart_centers is provided, a fraction of points are biased
    around those centers. The rest are uniform Sobol.

    Returns:
        (n_points, 7) array in original scale
    """
    torch.manual_seed(seed)

    if warmstart_centers is not None and len(warmstart_centers) > 0:
        n_warm = int(n_points * warmstart_frac)
        n_sobol = n_points - n_warm

        # Uniform Sobol portion
        sobol_norm = draw_sobol_samples(
            bounds=torch.tensor([[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=DTYPE),
            n=n_sobol, q=1, seed=seed,
        ).squeeze(1)
        X_sobol = unnormalize(sobol_norm, BOUNDS).cpu().numpy()

        centers = [np.asarray(c) for c in warmstart_centers]
        rng = np.random.RandomState(seed + 1)
        X_warm = np.empty((n_warm, N_DIMS))
        for i in range(n_warm):
            center = centers[i % len(centers)]
            noise = rng.uniform(-0.15, 0.15, size=N_DIMS)
            ranges = BOUNDS_UPPER - BOUNDS_LOWER
            x_new = center + noise * ranges
            x_new = np.clip(x_new, BOUNDS_LOWER, BOUNDS_UPPER)
            X_warm[i] = x_new

        X = np.vstack([X_sobol, X_warm])
        rng.shuffle(X)
        return X
    else:
        sobol_norm = draw_sobol_samples(
            bounds=torch.tensor([[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=DTYPE),
            n=n_points, q=1, seed=seed,
        ).squeeze(1)
        return unnormalize(sobol_norm, BOUNDS).cpu().numpy()


# ---------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------

def check_feasibility(constraints: dict) -> tuple[bool, dict]:
    """Check if a design point satisfies all constraints."""
    violations = {}
    if constraints["K"] <= CONSTRAINTS["K_min"]:
        violations["K"] = constraints["K"] - CONSTRAINTS["K_min"]
    if constraints["B1"] <= CONSTRAINTS["B1_min"]:
        violations["B1"] = constraints["B1"] - CONSTRAINTS["B1_min"]
    if constraints["S11_dB"] > CONSTRAINTS["S11_max_dB"]:
        violations["S11"] = constraints["S11_dB"] - CONSTRAINTS["S11_max_dB"]
    if constraints["S22_dB"] > CONSTRAINTS["S22_max_dB"]:
        violations["S22"] = constraints["S22_dB"] - CONSTRAINTS["S22_max_dB"]
    if constraints["OIP3_dBm"] < CONSTRAINTS["OIP3_min_dBm"]:
        violations["OIP3"] = constraints["OIP3_dBm"] - CONSTRAINTS["OIP3_min_dBm"]
    if constraints["Pdc_mW"] > CONSTRAINTS["Pdc_max_mW"]:
        violations["Pdc"] = constraints["Pdc_mW"] - CONSTRAINTS["Pdc_max_mW"]
    return len(violations) == 0, violations


# ---------------------------------------------------------------
# DKL: Feature extractor as BoTorch InputTransform
# ---------------------------------------------------------------

class DKLFeatureExtractor(botorch.models.transforms.input.InputTransform, torch.nn.Module):
    """NN-based input transform for Deep Kernel Learning."""

    def __init__(self, input_dim: int = N_DIMS, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True

    def transform(self, X):
        return self.net(X.to(self.net[0].weight.dtype))

    def forward(self, X):
        return self.transform(X)


# ---------------------------------------------------------------
# Model building
# ---------------------------------------------------------------

def build_model_list(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    use_warping: bool = False,
    use_dkl: bool = False,
) -> tuple[ModelListGP, SumMarginalLogLikelihood]:
    """Build and fit GP model list.

    Args:
        train_X: (n, 7) normalized inputs in [0, 1]^7
        train_Y: (n, m) targets. m = N_OBJECTIVES + N_CONSTRAINTS
        use_warping: if True, apply Kumaraswamy input warping
        use_dkl: if True, use Deep Kernel Learning

    Returns:
        Fitted ModelListGP and its SumMarginalLogLikelihood.
    """
    n_outputs = train_Y.shape[-1]

    shared_fe = None
    if use_dkl:
        shared_fe = _train_feature_extractor(train_X, train_Y, n_outputs)

    input_warp = None
    if use_warping:
        from input_warping import get_input_warp
        input_warp = get_input_warp(dim=N_DIMS)

    models = []
    for i in range(n_outputs):
        y_i = train_Y[..., i:i + 1]
        kwargs = {"outcome_transform": Standardize(m=1)}

        if shared_fe is not None:
            kwargs["input_transform"] = shared_fe
            kwargs["covar_module"] = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=shared_fe.output_dim)
            )
        elif input_warp is not None:
            # Each GP gets its own copy of the warp
            from input_warping import get_input_warp
            kwargs["input_transform"] = get_input_warp(dim=N_DIMS)

        model = SingleTaskGP(train_X, y_i, **kwargs)
        models.append(model)

    model_list = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    fit_gpytorch_mll(mll)

    return model_list, mll


def _train_feature_extractor(
    train_X: torch.Tensor, train_Y: torch.Tensor, n_outputs: int,
    n_epochs: int = 100, lr: float = 1e-3,
) -> DKLFeatureExtractor:
    """Pre-train shared NN for DKL."""
    fe = DKLFeatureExtractor(
        input_dim=N_DIMS, hidden_dim=32, output_dim=3,
    ).to(dtype=DTYPE, device=DEVICE)

    head = torch.nn.Linear(fe.output_dim, n_outputs).to(dtype=DTYPE, device=DEVICE)

    y_mean = train_Y.mean(0)
    y_std = train_Y.std(0).clamp(min=1e-8)
    Y_std = (train_Y - y_mean) / y_std

    optimizer = torch.optim.Adam(
        list(fe.parameters()) + list(head.parameters()), lr=lr,
    )

    fe.train()
    head.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        features = fe(train_X)
        pred = head(features)
        loss = torch.nn.functional.mse_loss(pred, Y_std)
        loss.backward()
        optimizer.step()

    fe.eval()
    for p in fe.parameters():
        p.requires_grad_(False)

    return fe


# ---------------------------------------------------------------
# Acquisition function (constraint-aware via callables)
# ---------------------------------------------------------------

def get_candidates(
    model: ModelListGP,
    train_X: torch.Tensor,
    ref_point: torch.Tensor,
    bounds_norm: torch.Tensor,
    config: OptimizationConfig,
    acr_slack: list[float] | None = None,
) -> torch.Tensor:
    """Optimize qLogNEHVI acquisition to get next candidate(s).

    Objectives are columns [0:3], constraints are columns [3:9].
    Constraint callables: c(samples) >= 0 means feasible.

    Args:
        acr_slack: per-constraint relaxation slack. If provided, each
            constraint callable returns Y[..., idx] + slack[i], effectively
            lowering the feasibility bar so the acqf explores more broadly.
            The GP model is unaffected — only the acqf decision boundary shifts.
    """
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([config.mc_samples])
    )

    objective = IdentityMCMultiOutputObjective(
        outcomes=list(range(N_OBJECTIVES))
    )

    if acr_slack is not None:
        # Relaxed constraints: c(Y) = Y[..., idx] + slack >= 0
        constraints = [
            lambda Y, idx=N_OBJECTIVES + i, s=acr_slack[i]: Y[..., idx] + s
            for i in range(N_CONSTRAINTS)
        ]
    else:
        # Standard constraints: c(Y) = Y[..., idx] >= 0
        constraints = [
            lambda Y, idx=idx: Y[..., idx]
            for idx in range(N_OBJECTIVES, N_OBJECTIVES + N_CONSTRAINTS)
        ]

    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=train_X,
        sampler=sampler,
        objective=objective,
        constraints=constraints,
        prune_baseline=True,
        cache_root=False,
    )

    candidates, acqf_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_norm,
        q=config.batch_size,
        num_restarts=config.num_restarts,
        raw_samples=config.raw_samples,
        options={"batch_limit": 5, "maxiter": 100},
        sequential=True,
    )

    logger.info("qLogNEHVI acquisition value: %.4f", acqf_value.sum().item())
    return candidates


# ---------------------------------------------------------------
# Main BO loop
# ---------------------------------------------------------------

def run_bo(
    n_initial: int = 20,
    n_iterations: int = 40,
    q: int = 3,
    seed: int = 42,
    log_dir: str | None = None,
    resume_from: str | None = None,
    pid_tag: str = "",
    use_warping: bool = False,
    use_dkl: bool = False,
    use_relaxation: bool = False,
    num_restarts: int = 10,
    raw_samples: int = 512,
    warmstart_centers: list | None = None,
    seed_results: list | None = None,
):
    """Run the full BO loop.

    Args:
        n_initial: number of initial Sobol samples
        n_iterations: number of BO iterations
        q: batch size
        seed: random seed
        log_dir: directory to save logs
        resume_from: path to existing all_results.json to resume
        use_warping: enable Kumaraswamy input warping
        use_dkl: enable Deep Kernel Learning
        use_relaxation: enable Adaptive Constraint Relaxation (ACR)
        warmstart_centers: centers for biased DoE sampling
        seed_results: pre-evaluated results to skip DoE phase
    """
    config = OptimizationConfig(
        n_initial=n_initial,
        n_iterations=n_iterations,
        batch_size=q,
        seed=seed,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    if log_dir is None:
        log_dir = os.path.join(LOCAL_RESULTS_DIR, f"run_s{seed}")
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    config_dict = {
        "n_initial": n_initial,
        "n_iterations": n_iterations,
        "batch_size": q,
        "seed": seed,
        "use_warping": use_warping,
        "use_dkl": use_dkl,
        "use_relaxation": use_relaxation,
        "n_dims": N_DIMS,
        "dim_names": DIM_NAMES,
        "num_restarts": config.num_restarts,
        "raw_samples": config.raw_samples,
        "mc_samples": config.mc_samples,
        "constraints": dict(CONSTRAINTS),
        "bounds_lower": BOUNDS_LOWER.tolist(),
        "bounds_upper": BOUNDS_UPPER.tolist(),
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    pid_prefix = f"{pid_tag}_s{seed}" if pid_tag else f"s{seed}"

    print("=" * 60)
    print(f"  PA Direct Geometry MOBO — Seed: {seed}")
    print(f"  DoE: {n_initial}, Iterations: {n_iterations}, q={q}")
    print(f"  Total evaluations: {n_initial + n_iterations * q}")
    print(f"  Dims: {N_DIMS} (all continuous)")
    print(f"  Warping: {use_warping}, DKL: {use_dkl}, ACR: {use_relaxation}")
    print(f"  Point ID prefix: {pid_prefix}")
    print("=" * 60)

    # ============================================================
    # Phase 1: Initial DoE (or resume / seed)
    # ============================================================
    if resume_from is not None:
        print(f"\n[Resume] Loading from {resume_from}...")
        with open(resume_from, "r") as f:
            all_results = json.load(f)
        completed_iters = max(0, (len(all_results) - n_initial) // q)
        print(f"  Loaded {len(all_results)} points ({completed_iters} BO iters)")
        for r in all_results:
            if "constraints" in r:
                feasible, viol = check_feasibility(r["constraints"])
                r["feasible"] = feasible
                r["violations"] = viol
        n_feas = sum(1 for r in all_results if r.get("feasible", False))
        print(f"  Feasible (re-evaluated): {n_feas}")
    elif seed_results is not None:
        import copy
        all_results = copy.deepcopy(seed_results)
        for r in all_results:
            if "constraints" in r:
                feasible, viol = check_feasibility(r["constraints"])
                r["feasible"] = feasible
                r["violations"] = viol
        n_feas = sum(1 for r in all_results if r.get("feasible", False))
        print(f"\n[Seed] Using {len(all_results)} pre-evaluated points "
              f"({n_feas} feasible)")
        _save_results(all_results, os.path.join(log_dir, "seed_results.json"))
        completed_iters = 0
    else:
        print(f"\n[Phase 1] Running initial DoE ({n_initial} points)...")
        X_init = generate_initial_doe(
            n_initial, seed,
            warmstart_frac=config.warmstart_frac,
            warmstart_centers=warmstart_centers,
        )

        all_results = []
        for i in range(n_initial):
            point_id = f"{pid_prefix}_doe_{i:03d}"
            print(f"  DoE {i+1}/{n_initial}: ", end="", flush=True)
            result = evaluate_point(X_init[i], point_id)

            if result.get("rejected"):
                result["feasible"] = False
                result["violations"] = {"d_in": "geometry rejected"}
            else:
                feasible, viol = check_feasibility(result["constraints"])
                result["feasible"] = feasible
                result["violations"] = viol

            all_results.append(result)
            obj = result["objectives"]
            feas_str = "FEAS" if result.get("feasible") else "INFEAS"
            if result.get("rejected"):
                feas_str = "REJECTED"
            print(f"PAE={obj['PAE_pct']:.1f}% "
                  f"Gain={obj['Gain_dB']:.1f}dB "
                  f"Psat={obj['Psat_dBm']:.1f}dBm "
                  f"{feas_str} ({result['sim_time_s']:.1f}s)")

        _save_results(all_results, os.path.join(log_dir, "doe_results.json"))
        completed_iters = 0

    # ============================================================
    # Compute reference point from DoE
    # ============================================================
    obj_matrix = np.array([
        [r["objectives"][k] for k in OBJ_NAMES] for r in all_results
    ])
    ref_point = []
    for j in range(N_OBJECTIVES):
        col = obj_matrix[:, j]
        rng_val = col.max() - col.min()
        ref_point.append(col.min() - 0.1 * max(rng_val, 1e-6))
    ref_point = torch.tensor(ref_point, dtype=DTYPE, device=DEVICE)
    print(f"\n  Reference point: PAE={ref_point[0]:.2f}%, "
          f"Gain={ref_point[1]:.2f}dB, Psat={ref_point[2]:.2f}dBm")

    # ============================================================
    # Phase 2: BO iterations
    # ============================================================
    remaining = n_iterations - completed_iters
    print(f"\n[Phase 2] Starting BO ({remaining} iterations, q={q})...")

    hv_history = []
    bounds_norm = torch.tensor(
        [[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=DTYPE, device=DEVICE
    )

    for iteration in range(completed_iters, n_iterations):
        t_iter_start = time.time()

        # Collect training data (skip rejected points for GP fitting)
        valid_results = [r for r in all_results if not r.get("rejected")]

        X_all = np.array([r["x"] for r in valid_results])
        Y_obj = np.array([
            [r["objectives"][k] for k in OBJ_NAMES] for r in valid_results
        ])
        Y_con_raw = np.array([
            [r["constraints"][name] for name, _, _ in CONSTRAINT_TRANSFORMS]
            for r in valid_results
        ])

        # Transform constraints: scale * raw + offset, so >= 0 is feasible
        Y_con = np.empty_like(Y_con_raw)
        for ci, (_, scale, offset) in enumerate(CONSTRAINT_TRANSFORMS):
            Y_con[:, ci] = Y_con_raw[:, ci] * scale + offset

        Y_all = np.hstack([Y_obj, Y_con])

        train_X = torch.tensor(X_all, dtype=DTYPE, device=DEVICE)
        train_Y = torch.tensor(Y_all, dtype=DTYPE, device=DEVICE)
        train_X_norm = normalize(train_X, BOUNDS)

        # Fit GPs (3 obj + 6 constraints)
        model_list, mll = build_model_list(
            train_X_norm, train_Y,
            use_warping=use_warping,
            use_dkl=use_dkl,
        )

        # Compute hypervolume (feasible points only)
        feasible_mask = torch.tensor(
            [r.get("feasible", False) for r in all_results], dtype=torch.bool
        )
        if feasible_mask.any():
            all_obj = np.array([
                [r["objectives"][k] for k in OBJ_NAMES] for r in all_results
            ])
            feasible_Y = torch.tensor(all_obj, dtype=DTYPE)[feasible_mask]
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_point, Y=feasible_Y,
            )
            hv = partitioning.compute_hypervolume().item()
        else:
            hv = 0.0
        hv_history.append(hv)

        n_total = len(all_results)
        n_feas = int(feasible_mask.sum())
        n_rejected = sum(1 for r in all_results if r.get("rejected"))
        print(f"\n  Iter {iteration+1}/{n_iterations} | "
              f"HV={hv:.4f} | "
              f"Data: {n_total} pts ({n_feas} feas, {n_rejected} rejected)")

        # Compute ACR slack for this iteration (if enabled)
        acr_slack = None
        if use_relaxation:
            acr_slack = get_acr_slack(iteration, n_iterations)
            slack_str = " ".join(f"{s:.2f}" for s in acr_slack)
            print(f"    ACR slack: [{slack_str}]")

        # Get candidates via qLogNEHVI
        candidates_norm = get_candidates(
            model_list, train_X_norm, ref_point, bounds_norm, config,
            acr_slack=acr_slack,
        )

        # Unnormalize to original scale
        candidates = unnormalize(candidates_norm, BOUNDS).cpu().numpy()

        # Evaluate batch
        batch_results = evaluate_batch(candidates, batch_id=iteration, pid_prefix=pid_prefix)

        for r in batch_results:
            if r.get("rejected"):
                r["feasible"] = False
                r["violations"] = {"d_in": "geometry rejected"}
            else:
                feasible, viol = check_feasibility(r["constraints"])
                r["feasible"] = feasible
                r["violations"] = viol

            all_results.append(r)
            obj = r["objectives"]
            feas_str = "FEAS" if r.get("feasible") else "INFEAS"
            if r.get("rejected"):
                feas_str = "REJECTED"
            print(f"    {r['point_id']}: "
                  f"PAE={obj['PAE_pct']:.1f}% "
                  f"Gain={obj['Gain_dB']:.1f}dB "
                  f"Psat={obj['Psat_dBm']:.1f}dBm "
                  f"{feas_str} ({r['sim_time_s']:.1f}s)")

        t_iter = time.time() - t_iter_start
        print(f"    Iteration time: {t_iter:.1f}s")

        # Checkpoint
        _save_results(all_results, os.path.join(log_dir, "all_results.json"))
        _save_hv(hv_history, os.path.join(log_dir, "hv_history.json"))

    # ============================================================
    # Final summary
    # ============================================================
    _print_summary(all_results, ref_point, log_dir)


def _print_summary(all_results, ref_point, log_dir):
    """Print final optimization summary."""
    print("\n" + "=" * 60)
    print("  Optimization Complete — PA Direct Geometry")
    print("=" * 60)

    valid = [r for r in all_results if not r.get("rejected")]
    Y_final = torch.tensor(
        [[r["objectives"][k] for k in OBJ_NAMES] for r in valid],
        dtype=DTYPE,
    )
    feas_final = torch.tensor(
        [r.get("feasible", False) for r in valid], dtype=torch.bool
    )

    n_rejected = sum(1 for r in all_results if r.get("rejected"))

    if feas_final.any():
        feas_Y = Y_final[feas_final]
        part = FastNondominatedPartitioning(ref_point=ref_point, Y=feas_Y)
        final_hv = part.compute_hypervolume().item()
        print(f"  Final HV: {final_hv:.4f}")
        print(f"  Total: {len(all_results)} | Valid: {len(valid)} | "
              f"Feasible: {feas_final.sum()} | Rejected (d_in): {n_rejected}")

        pareto_mask = is_non_dominated(feas_Y)
        print(f"  Pareto front size: {pareto_mask.sum()}")
        print(f"\n  Pareto front designs:")

        feas_indices = [i for i, r in enumerate(valid) if r.get("feasible")]
        pareto_global = [feas_indices[j] for j in range(len(feas_indices)) if pareto_mask[j]]
        for idx in pareto_global:
            r = valid[idx]
            print(f"    {r['point_id']}: "
                  f"PAE={r['objectives']['PAE_pct']:.2f}% "
                  f"Gain={r['objectives']['Gain_dB']:.2f}dB "
                  f"Psat={r['objectives']['Psat_dBm']:.2f}dBm "
                  f"| L1L2: rad={r['L1L2']['rad']:.1f}um nr={r['L1L2']['nr']:.1f} "
                  f"| L3L4: rad={r['L3L4']['rad']:.1f}um nr={r['L3L4']['nr']:.1f}")
    else:
        print(f"  No feasible designs found! "
              f"(Total: {len(all_results)}, Rejected: {n_rejected})")

    _save_results(all_results, os.path.join(log_dir, "all_results.json"))
    print(f"\n  Results saved to: {log_dir}")


# ---------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------

def _save_results(results: list, filepath: str):
    """Save results to JSON (converting numpy types)."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    clean = []
    for r in results:
        r_clean = {}
        for k, v in r.items():
            if k == "raw":
                r_raw = {}
                for rk, rv in v.items():
                    if rk == "hb":
                        r_raw[rk] = {
                            kk: vv for kk, vv in rv.items()
                            if kk in ["PAE_pct", "Gain_dB", "Psat_dBm", "error"]
                        }
                    else:
                        r_raw[rk] = rv
                r_clean[k] = r_raw
            else:
                r_clean[k] = v
        clean.append(r_clean)

    with open(filepath, "w") as f:
        json.dump(convert(clean), f, indent=2)


def _save_hv(hv_history: list, filepath: str):
    """Save HV history."""
    with open(filepath, "w") as f:
        json.dump(hv_history, f, indent=2)


if __name__ == "__main__":
    run_bo(
        n_initial=20,
        n_iterations=40,
        q=3,
        seed=42,
    )

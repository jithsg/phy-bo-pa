"""Geometry-Aware MOBO for PA Direct Geometry Optimization.

Extends bo_loop.py with three key novelties:

1. **Analytical d_in constraints**: The inner-diameter validity check
   (d_in = 2*rad - 2*nr*(w+sp) >= D_IN_MIN) is embedded as a differentiable
   linear inequality constraint in optimize_acqf. The optimizer *never*
   proposes geometrically invalid points, eliminating ~14% budget waste.

2. **PDK grid-aware snapping**: Inductor rad (integer um) and nr (half-turn)
   are snapped *before* simulation, and the snapped X is stored in GP training
   data. This closes the train-eval gap where the GP previously saw continuous
   proposals but outcomes corresponded to snapped geometries.

3. **Physics-informed GP mean functions**: For outputs with known analytical
   models (Gain, S11, S22, Pdc), the GP prior mean is set to the physics
   prediction. The GP then only needs to learn the residual, improving
   sample efficiency especially for the binding S11 constraint.

4. **Geometry-filtered DoE**: Initial DoE points are reject-sampled to
   guarantee d_in validity, so no DoE budget is wasted either.

Usage:
    from bo_loop_geo import run_bo  # drop-in replacement
"""

import os
import json
import time
import logging
import math
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
from physics_models_geo import (
    create_objective_physics_means,
    create_constraint_physics_means,
    OBJECTIVE_PHYSICS,
    CONSTRAINT_PHYSICS,
    PhysicsMean,
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from config import (
    BOUNDS_LOWER, BOUNDS_UPPER, CONSTRAINTS, LOCAL_RESULTS_DIR,
    DIM_NAMES, N_DIMS, PRESIM_X, PRESIM_OBJECTIVES,
    IND_W, IND_SPACING, D_IN_MIN_UM,
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
# Pdc uses tightened threshold (24.5 mW) to absorb PDK grid-snap margin
_PDC_SNAP_MARGIN = 0.5  # mW safety margin for grid snapping
CONSTRAINT_TRANSFORMS = [
    ("K",        1.0,  -CONSTRAINTS["K_min"]),
    ("B1",       1.0,  -CONSTRAINTS["B1_min"]),
    ("S11_dB",  -1.0,   CONSTRAINTS["S11_max_dB"]),
    ("S22_dB",  -1.0,   CONSTRAINTS["S22_max_dB"]),
    ("OIP3_dBm", 1.0,  -CONSTRAINTS["OIP3_min_dBm"]),
    ("Pdc_mW",  -1.0,   CONSTRAINTS["Pdc_max_mW"] - _PDC_SNAP_MARGIN),
]
N_CONSTRAINTS = len(CONSTRAINT_TRANSFORMS)
N_OBJECTIVES = 3



# ---------------------------------------------------------------
# Geometry-aware: analytical d_in constraints in normalized space
# ---------------------------------------------------------------

def _make_din_linear_constraint(rad_idx: int, nr_idx: int):
    """Create a linear d_in >= D_IN_MIN constraint for optimize_acqf.

    d_in = 2*rad - 2*nr*(w + sp)  [all in um]

    In normalized coords x ∈ [0,1]:
        rad = x[rad_idx] * (rad_max - rad_min) + rad_min
        nr  = x[nr_idx]  * (nr_max  - nr_min)  + nr_min

    Returns (indices, coefficients, rhs) tuple for BoTorch's
    inequality_constraints: coefficients @ x[indices] >= rhs.
    """
    rad_lo, rad_hi = float(BOUNDS_LOWER[rad_idx]), float(BOUNDS_UPPER[rad_idx])
    nr_lo, nr_hi = float(BOUNDS_LOWER[nr_idx]), float(BOUNDS_UPPER[nr_idx])
    w_sp = IND_W + IND_SPACING

    # d_in = a_rad * x_rad + a_nr * x_nr + c >= 0
    # Rewrite as: a_rad * x_rad + a_nr * x_nr >= -c  (BoTorch format)
    #
    # Snap margin: absorbs the d_in shift caused by rounding rad to the
    # PDK radius grid or nr to half-turns (see paper, Sec. III-E).
    SNAP_MARGIN = 1.5

    a_rad = 2.0 * (rad_hi - rad_lo)
    a_nr = -2.0 * (nr_hi - nr_lo) * w_sp
    c = 2.0 * rad_lo - 2.0 * nr_lo * w_sp - D_IN_MIN_UM - SNAP_MARGIN
    # a_rad * x_rad + a_nr * x_nr >= -c

    indices = torch.tensor([rad_idx, nr_idx], dtype=torch.long)
    coefficients = torch.tensor([a_rad, a_nr], dtype=DTYPE)
    rhs = -c

    return (indices, coefficients, rhs)


# Two linear constraints: L1L2 (dims 0,1) and L3L4 (dims 2,3)
DIN_LINEAR_CONSTRAINTS = [
    _make_din_linear_constraint(0, 1),
    _make_din_linear_constraint(2, 3),
]


# ---------------------------------------------------------------
# Geometry-aware: PDK grid snapping
# ---------------------------------------------------------------

def snap_to_pdk_grid(x: np.ndarray) -> np.ndarray:
    """Snap inductor params to PDK grid, in-place on a copy.

    - rad (dims 0, 2): round to nearest integer um
    - nr  (dims 1, 3): round to nearest 0.5 turns

    Other dims (C_wr, R_l, M_wr) are continuous — no snapping.
    """
    x_snapped = x.copy()
    # L1L2_rad, L3L4_rad → integer
    x_snapped[..., 0] = np.round(x_snapped[..., 0])
    x_snapped[..., 2] = np.round(x_snapped[..., 2])
    # L1L2_nr, L3L4_nr → half-integer
    x_snapped[..., 1] = np.round(x_snapped[..., 1] * 2) / 2
    x_snapped[..., 3] = np.round(x_snapped[..., 3] * 2) / 2
    # Clip to bounds after snapping
    x_snapped = np.clip(x_snapped, BOUNDS_LOWER, BOUNDS_UPPER)
    return x_snapped


def check_din_valid(x: np.ndarray) -> bool:
    """Check if a single point (7,) passes both d_in checks."""
    d1 = 2 * x[0] - 2 * x[1] * (IND_W + IND_SPACING)
    d2 = 2 * x[2] - 2 * x[3] * (IND_W + IND_SPACING)
    return d1 >= D_IN_MIN_UM and d2 >= D_IN_MIN_UM


# ---------------------------------------------------------------
# Geometry-filtered DoE generation
# ---------------------------------------------------------------

def generate_initial_doe(
    n_points: int,
    seed: int = 42,
    warmstart_frac: float = 0.5,
    warmstart_centers: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Generate initial DoE with geometry-validity guarantee.

    All returned points satisfy d_in >= D_IN_MIN after PDK snapping.
    Uses reject-sampling: generate candidates, snap, filter, repeat.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    collected = []
    oversample = 3  # generate 3x to account for rejections
    attempt = 0

    while len(collected) < n_points and attempt < 20:
        attempt += 1
        n_need = (n_points - len(collected)) * oversample

        if warmstart_centers is not None and len(warmstart_centers) > 0:
            n_warm = int(n_need * warmstart_frac)
            n_sobol = n_need - n_warm

            sobol_norm = draw_sobol_samples(
                bounds=torch.tensor([[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=DTYPE),
                n=n_sobol, q=1, seed=seed + attempt,
            ).squeeze(1)
            X_sobol = unnormalize(sobol_norm, BOUNDS).cpu().numpy()

            centers = [np.asarray(c) for c in warmstart_centers]
            X_warm = np.empty((n_warm, N_DIMS))
            for i in range(n_warm):
                center = centers[i % len(centers)]
                noise = rng.uniform(-0.15, 0.15, size=N_DIMS)
                ranges = BOUNDS_UPPER - BOUNDS_LOWER
                x_new = center + noise * ranges
                x_new = np.clip(x_new, BOUNDS_LOWER, BOUNDS_UPPER)
                X_warm[i] = x_new

            X_raw = np.vstack([X_sobol, X_warm])
        else:
            sobol_norm = draw_sobol_samples(
                bounds=torch.tensor([[0.0] * N_DIMS, [1.0] * N_DIMS], dtype=DTYPE),
                n=n_need, q=1, seed=seed + attempt,
            ).squeeze(1)
            X_raw = unnormalize(sobol_norm, BOUNDS).cpu().numpy()

        # Snap and filter
        for i in range(X_raw.shape[0]):
            x_snap = snap_to_pdk_grid(X_raw[i])
            if check_din_valid(x_snap):
                collected.append(x_snap)
                if len(collected) >= n_points:
                    break

    X = np.array(collected[:n_points])
    rng.shuffle(X)
    return X


# ---------------------------------------------------------------
# Feasibility check (same as bo_loop.py)
# ---------------------------------------------------------------

def check_feasibility(constraints: dict) -> tuple[bool, dict]:
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
# Physics-derived input features (appended to X for GP)
# ---------------------------------------------------------------
# Key insight: S11 depends on resonance match ωL1 ≈ 1/(ωCgs).
# Instead of a mean function (which was 6dB off), we give the GP
# direct access to these computed features so it can learn the
# S11/S22 relationship itself.

_F0 = 2.4e9
_OMEGA = 2.0 * math.pi * _F0
_MU0_LOCAL = 4.0 * math.pi * 1e-7
_WHEELER_K1_LOCAL = 1.754
_WHEELER_K2_LOCAL = -0.410
_W_SP = (IND_W + IND_SPACING) * 1e-6  # m


class PhysicsFeatureTransform(botorch.models.transforms.input.InputTransform, torch.nn.Module):
    """Appends physics-derived features to input X.

    Input:  X of shape (..., 7) — normalized [0,1]
    Output: X of shape (..., 7 + n_features)

    Features appended:
      - X_L1:  ωL1 (input inductive reactance, Ohms)
      - X_Cgs: 1/(ωCgs) (gate capacitive reactance, Ohms)
      - mismatch_in:  (X_L1 - X_Cgs) / Z0  (normalized input mismatch)
      - mismatch_out: (X_L3 - X_Cds) / Z0  (normalized output mismatch)
    """

    def __init__(self, bounds: torch.Tensor):
        super().__init__()
        self.register_buffer("bounds", bounds)
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True

    def _wheeler_L(self, rad_um: torch.Tensor, nr: torch.Tensor) -> torch.Tensor:
        """Wheeler inductance in Henrys."""
        d_out = 2.0 * rad_um * 1e-6
        d_in = d_out - 2.0 * nr * _W_SP
        d_in = torch.clamp(d_in, min=1e-6)
        d_avg = (d_out + d_in) / 2.0
        rho = (d_out - d_in) / (d_out + d_in + 1e-12)
        L = _WHEELER_K1_LOCAL * _MU0_LOCAL * nr ** 2 * d_avg / (1.0 + _WHEELER_K2_LOCAL * rho)
        return L

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # Unnormalize to raw units
        x_raw = unnormalize(X, self.bounds)

        rad1 = x_raw[..., 0]  # L1L2 rad (um)
        nr1 = x_raw[..., 1]   # L1L2 nr
        rad3 = x_raw[..., 2]  # L3L4 rad (um)
        nr3 = x_raw[..., 3]   # L3L4 nr
        m_wr = x_raw[..., 6]  # MOS width (m)

        # Input inductor reactance
        L1 = self._wheeler_L(rad1, nr1)
        X_L1 = _OMEGA * L1

        # Gate capacitance reactance
        W_total = m_wr * 100  # mos_multi=100
        Cgs = 8.0 * ((2.0 / 3.0) * 10e-3 * W_total * 40e-9 + 0.3e-6 * W_total)
        X_Cgs = 1.0 / (_OMEGA * torch.clamp(Cgs, min=1e-15))

        # Output inductor reactance
        L3 = self._wheeler_L(rad3, nr3)
        X_L3 = _OMEGA * L3

        # Drain capacitance reactance
        Cds = 25.0 * 0.2e-6 * W_total
        X_Cds = 1.0 / (_OMEGA * torch.clamp(Cds, min=1e-15))

        # Physics-derived matching scores
        # S11 matching in this push-pull PA is NOT simple LC resonance.
        # Empirical finding (corr=-0.72 with S11):
        #   match_in = ω²·L1·Cgs·c_wr/r_l
        # Higher → better S11 (more impedance transformation).
        c_wr = x_raw[..., 4]  # coupling cap width (m)
        r_l = x_raw[..., 5]   # load resistance width (m)

        match_in = _OMEGA ** 2 * L1 * Cgs * c_wr / torch.clamp(r_l, min=1e-9)
        match_out = _OMEGA ** 2 * L3 * Cds

        # Normalize to [0, 1] via log scale
        # match_in range: ~[300, 4000] → log10: [2.5, 3.6]
        match_in_norm = (torch.log10(torch.clamp(match_in, min=1.0)) - 2.5) / 1.5
        # match_out range: ~[500, 3000] → log10: [2.7, 3.5]
        match_out_norm = (torch.log10(torch.clamp(match_out, min=1.0)) - 2.7) / 0.8

        # Inductor fill factors (affects Q and loss)
        d_out1 = 2.0 * rad1 * 1e-6
        d_in1 = d_out1 - 2.0 * nr1 * _W_SP
        rho1 = (d_out1 - torch.clamp(d_in1, min=1e-6)) / (d_out1 + torch.clamp(d_in1, min=1e-6) + 1e-12)

        d_out3 = 2.0 * rad3 * 1e-6
        d_in3 = d_out3 - 2.0 * nr3 * _W_SP
        rho3 = (d_out3 - torch.clamp(d_in3, min=1e-6)) / (d_out3 + torch.clamp(d_in3, min=1e-6) + 1e-12)

        features = torch.stack([
            match_in_norm,   # input matching score [~0, ~1] (higher → better S11)
            match_out_norm,  # output matching score [~0, ~1]
            rho1,            # input inductor fill factor [0, 1]
            rho3,            # output inductor fill factor [0, 1]
        ], dim=-1)

        # Clamp to prevent outliers from blowing up kernel
        features = torch.clamp(features, min=-0.5, max=1.5)

        return torch.cat([X, features], dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)


N_PHYSICS_FEATURES = 4  # number of appended features


# ---------------------------------------------------------------
# DKL Feature extractor (same as bo_loop.py)
# ---------------------------------------------------------------

class DKLFeatureExtractor(botorch.models.transforms.input.InputTransform, torch.nn.Module):
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
# Model building — with optional physics-informed GP mean functions
# ---------------------------------------------------------------

def _build_physics_means() -> dict[int, PhysicsMean]:
    """Create physics mean functions for all outputs that have them.

    Returns dict mapping output index (in the 9-output Y tensor) to PhysicsMean.
    Objective indices: 0=PAE, 1=Gain, 2=Psat
    Constraint indices: 3=K, 4=B1, 5=S11, 6=S22, 7=OIP3, 8=Pdc
    """
    obj_means = create_objective_physics_means(BOUNDS)
    con_means = create_constraint_physics_means(BOUNDS, CONSTRAINT_TRANSFORMS)

    # Cheap-fidelity MLP priors (S11/S22/K/B1) supersede the empirical
    # regressions when the trained checkpoint is available; the constraint
    # transform maps the raw prediction into the GP target space exactly as
    # for the analytical priors.
    from mlp_prior import mlp_constraint_fns
    from physics_models_geo import PhysicsMean as _PM
    mlp_fns = mlp_constraint_fns()
    if mlp_fns is not None:
        for con_idx, fn in mlp_fns.items():
            _, scale, offset = CONSTRAINT_TRANSFORMS[con_idx]
            con_means[con_idx] = _PM(
                fn, BOUNDS,
                transform_fn=(lambda y, s=scale, o=offset: y * s + o))
        logger.info("MLP cheap-fidelity priors active for constraints %s",
                    sorted(mlp_fns))

    physics_means: dict[int, PhysicsMean] = {}
    for obj_idx, mean in obj_means.items():
        physics_means[obj_idx] = mean
    for con_idx, mean in con_means.items():
        physics_means[N_OBJECTIVES + con_idx] = mean

    return physics_means


def build_model_list(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    use_warping: bool = False,
    use_dkl: bool = False,
    use_physics: bool = True,
    use_physics_features: bool = True,
) -> tuple[ModelListGP, SumMarginalLogLikelihood]:
    n_outputs = train_Y.shape[-1]

    shared_fe = None
    if use_dkl:
        shared_fe = _train_feature_extractor(train_X, train_Y, n_outputs)

    input_warp = None
    if use_warping:
        from input_warping import get_input_warp
        input_warp = get_input_warp(dim=N_DIMS)

    physics_means = _build_physics_means() if use_physics else {}

    models = []
    for i in range(n_outputs):
        y_i = train_Y[..., i:i + 1]
        kwargs = {"outcome_transform": Standardize(m=1)}

        if i in physics_means:
            kwargs["mean_module"] = physics_means[i]

        if shared_fe is not None:
            kwargs["input_transform"] = shared_fe
            kwargs["covar_module"] = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=shared_fe.output_dim)
            )
        elif input_warp is not None:
            from input_warping import get_input_warp
            kwargs["input_transform"] = get_input_warp(dim=N_DIMS)
        elif use_physics_features:
            kwargs["input_transform"] = PhysicsFeatureTransform(BOUNDS)
            kwargs["covar_module"] = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=N_DIMS + N_PHYSICS_FEATURES)
            )

        model = SingleTaskGP(train_X, y_i, **kwargs)

        # Set standardization params on physics mean so it predicts in
        # the same space as the standardized GP targets
        if i in physics_means:
            ot = model.outcome_transform
            y_mean = ot.means.squeeze().item()
            y_std = ot.stdvs.squeeze().item()
            physics_means[i].set_standardization(y_mean, y_std)

        models.append(model)

    model_list = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        # ModelFittingError or numerical issues — use current hyperparams
        logger.warning("fit_gpytorch_mll failed (%s), using default hyperparams", e)

    return model_list, mll


def _train_feature_extractor(
    train_X: torch.Tensor, train_Y: torch.Tensor, n_outputs: int,
    n_epochs: int = 100, lr: float = 1e-3,
) -> DKLFeatureExtractor:
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
# Acquisition function — with analytical d_in constraints
# ---------------------------------------------------------------

def get_candidates(
    model: ModelListGP,
    train_X: torch.Tensor,
    ref_point: torch.Tensor,
    bounds_norm: torch.Tensor,
    config: OptimizationConfig,
    use_din_constraint: bool = True,
) -> torch.Tensor:
    """Optimize qLogNEHVI with analytical geometry constraints.

    Two layers of constraints:
    1. Model-based (soft): GP-predicted K, B1, S11, S22, OIP3, Pdc via callables
    2. Analytical (hard): d_in >= D_IN_MIN for both inductor pairs via
       linear inequality_constraints on the input space X.
       Disabled when use_din_constraint=False (constraint-only ablation arm:
       snapped BO whose acquisition optimizer is not told about d_in; invalid
       proposals are then caught post hoc by evaluate_point, exactly as in the
       non-geometry-aware Stage I).
    """
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([config.mc_samples])
    )

    objective = IdentityMCMultiOutputObjective(
        outcomes=list(range(N_OBJECTIVES))
    )

    # Standard model-based constraints: c(Y) = Y[..., idx] >= 0
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

    # Analytical d_in as linear inequality_constraints on input space X.
    # Format: list of (indices, coefficients, rhs) where coeff @ X[indices] >= rhs.
    # This ensures the optimizer never proposes geometrically invalid points.
    din_constraints = DIN_LINEAR_CONSTRAINTS if use_din_constraint else None
    try:
        candidates, acqf_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_norm,
            q=config.batch_size,
            num_restarts=config.num_restarts,
            raw_samples=config.raw_samples,
            options={"batch_limit": 5, "maxiter": 100},
            sequential=True,
            inequality_constraints=din_constraints,
        )
        logger.info("qLogNEHVI acquisition value: %.4f", acqf_value.sum().item())
    except Exception as e:
        # NaN gradient or other optimization failure — fall back to random candidate
        logger.warning("optimize_acqf failed (%s), using random candidate", e)
        candidates = torch.rand(config.batch_size, bounds_norm.shape[1],
                                dtype=DTYPE, device=DEVICE)
        # Ensure d_in validity for random fallback (only when the constraint is on)
        for _ in range(100 if use_din_constraint else 0):
            valid = True
            for idx, coeff, rhs in DIN_LINEAR_CONSTRAINTS:
                if (coeff * candidates[0, idx]).sum() < rhs:
                    valid = False
                    break
            if valid:
                break
            candidates = torch.rand(config.batch_size, bounds_norm.shape[1],
                                    dtype=DTYPE, device=DEVICE)

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
    use_physics: bool = True,
    use_physics_features: bool = False,
    num_restarts: int = 10,
    raw_samples: int = 512,
    warmstart_centers: list | None = None,
    seed_results: list | None = None,
    use_din_constraint: bool = True,
):
    """Run geometry-aware BO loop.

    Key differences from bo_loop.py:
    - DoE is geometry-filtered (no d_in rejections)
    - Candidates are snapped to PDK grid before evaluation
    - Snapped X is fed back to GP (closes train-eval gap)
    - Analytical d_in constraints in optimize_acqf (zero rejections during BO)
    - Physics-informed GP means for Gain, S11, S22, Pdc (modified Wheeler)
    - use_relaxation is accepted but ignored (ACR not used in this variant)
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
        "use_physics": use_physics,
        "use_physics_features": use_physics_features,
        "warmstart": warmstart_centers is not None,
        "resume_from": resume_from,
        "pid_tag": pid_tag,
        "geometry_aware": True,
        "use_din_constraint": use_din_constraint,
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
    print(f"  PA Direct Geometry MOBO — Geometry-Aware — Seed: {seed}")
    print(f"  DoE: {n_initial}, Iterations: {n_iterations}, q={q}")
    print(f"  Total evaluations: {n_initial + n_iterations * q}")
    print(f"  Dims: {N_DIMS} (all continuous, PDK-snapped)")
    print(f"  Warping: {use_warping}, DKL: {use_dkl}")
    if use_din_constraint:
        print(f"  Geometry constraints: d_in analytical + PDK grid snap")
    else:
        print(f"  Geometry constraints: PDK grid snap ONLY "
              f"(d_in inequality DISABLED — constraint-only ablation arm)")
    print(f"  Point ID prefix: {pid_prefix}")
    print("=" * 60)

    # ============================================================
    # Phase 1: Initial DoE (geometry-filtered)
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
        n_rej = sum(1 for r in all_results if r.get("rejected", False))
        print(f"  Feasible: {n_feas}, Rejected: {n_rej}")
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
        print(f"\n[Phase 1] Running geometry-filtered DoE ({n_initial} points)...")
        X_init = generate_initial_doe(
            n_initial, seed,
            warmstart_frac=config.warmstart_frac,
            warmstart_centers=warmstart_centers,
        )
        print(f"  All {n_initial} DoE points pass d_in check (geometry-filtered)")

        all_results = []
        for i in range(n_initial):
            point_id = f"{pid_prefix}_doe_{i:03d}"
            print(f"  DoE {i+1}/{n_initial}: ", end="", flush=True)
            result = evaluate_point(X_init[i], point_id)

            if result.get("rejected"):
                # Should not happen with geometry-filtered DoE
                print(f"WARNING: unexpected rejection in geometry-filtered DoE!")
                result["feasible"] = False
                result["violations"] = {"d_in": "geometry rejected"}
            else:
                feasible, viol = check_feasibility(result["constraints"])
                result["feasible"] = feasible
                result["violations"] = viol

            # Store the snapped X (already snapped by evaluate_point)
            all_results.append(result)
            obj = result["objectives"]
            feas_str = "FEAS" if result.get("feasible") else "INFEAS"
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
    # Phase 2: BO iterations (geometry-constrained)
    # ============================================================
    remaining = n_iterations - completed_iters
    print(f"\n[Phase 2] Starting geometry-aware BO ({remaining} iterations, q={q})...")

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

        # Fit GPs (3 obj + 6 constraints, with physics features + means)
        model_list, mll = build_model_list(
            train_X_norm, train_Y,
            use_warping=use_warping,
            use_dkl=use_dkl,
            use_physics=use_physics,
            use_physics_features=use_physics_features,
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

        # Get candidates via qLogNEHVI (with analytical d_in constraints,
        # unless disabled for the constraint-only ablation arm)
        candidates_norm = get_candidates(
            model_list, train_X_norm, ref_point, bounds_norm, config,
            use_din_constraint=use_din_constraint,
        )

        # Unnormalize to original scale
        candidates_raw = unnormalize(candidates_norm, BOUNDS).cpu().numpy()

        # PDK grid snap BEFORE evaluation
        candidates = np.array([snap_to_pdk_grid(candidates_raw[i])
                               for i in range(candidates_raw.shape[0])])

        # Log snap delta for diagnostics
        snap_delta = np.abs(candidates - candidates_raw).max(axis=1)
        for si in range(len(snap_delta)):
            if snap_delta[si] > 0.01:
                logger.info("  Snap delta pt%d: max=%.3f", si, snap_delta[si])

        # Evaluate batch
        batch_results = evaluate_batch(candidates, batch_id=iteration, pid_prefix=pid_prefix)

        for r in batch_results:
            if r.get("rejected"):
                # Should be extremely rare with analytical constraints;
                # expected occasionally in the unconstrained ablation arm.
                if use_din_constraint:
                    print(f"    WARNING: rejection despite analytical constraint! "
                          f"L1L2 d_in={r['L1L2']['d_in']:.1f}, "
                          f"L3L4 d_in={r['L3L4']['d_in']:.1f}")
                else:
                    print(f"    d_in rejection (unconstrained arm): "
                          f"L1L2 d_in={r['L1L2']['d_in']:.1f}, "
                          f"L3L4 d_in={r['L3L4']['d_in']:.1f}")
                r["feasible"] = False
                r["violations"] = {"d_in": "geometry rejected (post-snap)"}
            else:
                feasible, viol = check_feasibility(r["constraints"])
                r["feasible"] = feasible
                r["violations"] = viol

            all_results.append(r)
            obj = r["objectives"]
            feas_str = "FEAS" if r.get("feasible") else "INFEAS"
            if r.get("rejected"):
                feas_str = "REJECTED"
            s11_val = r["constraints"].get("S11_dB", 0)
            print(f"    {r['point_id']}: "
                  f"PAE={obj['PAE_pct']:.1f}% "
                  f"Gain={obj['Gain_dB']:.1f}dB "
                  f"Psat={obj['Psat_dBm']:.1f}dBm "
                  f"S11={s11_val:.1f}dB "
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
    print("\n" + "=" * 60)
    print("  Optimization Complete — PA Direct Geometry (Geo-Aware)")
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

    # Atomic write: write to temp file then rename to prevent corruption
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(convert(clean), f, indent=2)
    os.replace(tmp_path, filepath)


def _save_hv(hv_history: list, filepath: str):
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(hv_history, f, indent=2)
    os.replace(tmp_path, filepath)


if __name__ == "__main__":
    run_bo(
        n_initial=20,
        n_iterations=40,
        q=3,
        seed=42,
    )

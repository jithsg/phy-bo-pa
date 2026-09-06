"""Full MOBO run for PA direct geometry optimization.

7D continuous: [L1L2_rad, L1L2_nr, L3L4_rad, L3L4_nr, C_wr, R_l, M_wr]
3 objectives (PAE, Gain, Psat), 6 constraints.

Usage:
    python run_full.py                  # Standard GP
    python run_full.py --warping        # With input warping
    python run_full.py --warmstart      # With warm-start DoE
    python run_full.py --full           # Warping + warm-start
    python run_full.py --dkl            # Deep kernel learning
"""

import argparse
from config import OptimizationConfig, LOCAL_RESULTS_DIR, PRESIM_X
from bo_loop import run_bo


def main():
    parser = argparse.ArgumentParser(description="PA Direct Geometry MOBO")
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warping", action="store_true", help="Enable input warping")
    parser.add_argument("--warmstart", action="store_true", help="Bias DoE around PRESIM")
    parser.add_argument("--full", action="store_true", help="Warping + warm-start")
    parser.add_argument("--dkl", action="store_true", help="Deep kernel learning")
    parser.add_argument("--relaxation", action="store_true", help="Adaptive Constraint Relaxation")
    parser.add_argument("--resume", type=str, default=None, help="Resume from results JSON")
    parser.add_argument("--tag", type=str, default="geo", help="Run tag for point IDs")

    args = parser.parse_args()

    use_warping = args.warping or args.full
    use_warmstart = args.warmstart or args.full

    tag = args.tag
    if args.full:
        tag += "_full"
    elif args.warping:
        tag += "_warp"
    elif args.warmstart:
        tag += "_warm"
    else:
        tag += "_std"

    log_dir = f"{LOCAL_RESULTS_DIR}/{tag}_s{args.seed}"

    warmstart_centers = [PRESIM_X] if use_warmstart else None

    run_bo(
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        q=args.batch_size,
        seed=args.seed,
        log_dir=log_dir,
        use_warping=use_warping,
        use_dkl=args.dkl,
        use_relaxation=args.relaxation,
        warmstart_centers=warmstart_centers,
        resume_from=args.resume,
        pid_tag=tag,
    )


if __name__ == "__main__":
    main()

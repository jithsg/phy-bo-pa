"""Launch Geometry-Aware MOBO + live dashboard.

Same interface as run_with_dashboard.py but uses bo_loop_geo
(analytical d_in constraints + PDK grid snapping).

Usage:
    python run_geo_aware.py --warmstart --tag geo5 --batch-size 1 --n-iterations 80
    python run_geo_aware.py --warmstart --resume results/geo2_warm_s42/doe_results.json
"""

import argparse
import os
import threading
import time

from config import LOCAL_RESULTS_DIR, PRESIM_X, OptimizationConfig


def main():
    parser = argparse.ArgumentParser(description="PA Geo MOBO — Geometry-Aware")
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warping", action="store_true")
    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dkl", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tag", type=str, default="geo")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--raw-samples", type=int, default=256)
    parser.add_argument("--no-physics", action="store_true",
                        help="Disable physics-informed GP mean functions")
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
    tag += "_geoaware"

    log_dir = f"{LOCAL_RESULTS_DIR}/{tag}_s{args.seed}"
    os.makedirs(log_dir, exist_ok=True)

    warmstart_centers = [PRESIM_X] if use_warmstart else None

    # --- Start dashboard in background thread ---
    def run_dashboard():
        import dashboard
        dashboard.RESULTS_DIR = log_dir
        print(f"\n  Dashboard: http://localhost:{args.port}")
        print(f"  Watching: {log_dir}\n")
        dashboard.app.run(debug=False, port=args.port, host="0.0.0.0",
                          use_reloader=False)

    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()
    time.sleep(1)

    # --- Run geometry-aware BO in main thread ---
    from bo_loop_geo import run_bo

    run_bo(
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        q=args.batch_size,
        seed=args.seed,
        log_dir=log_dir,
        use_warping=use_warping,
        use_dkl=args.dkl,
        use_physics=not args.no_physics,
        warmstart_centers=warmstart_centers,
        resume_from=args.resume,
        pid_tag=tag,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
    )

    print(f"\nBO complete. Dashboard still running at http://localhost:{args.port}")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

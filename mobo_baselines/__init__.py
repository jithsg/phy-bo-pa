"""External MOBO/MOEA baselines for the PA direct-geometry optimization project.

All baselines reuse the existing ``evaluate.evaluate_point`` so the Spectre
flow, snapping, constraints, and netlist generation are bit-for-bit identical
to the proposed Base-BO (``bo_loop``) and Phys-BO (``bo_loop_geo``) runs.
Nothing in the parent project is modified; this package is purely additive.

Outputs are written to ``results/mobo_baselines/{name}_seed{seed}.json`` and
never touch the existing ``results/geo*`` artifacts.
"""

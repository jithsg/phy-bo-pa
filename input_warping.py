"""Kumaraswamy input warping for non-stationary GP modeling.

The Kumaraswamy CDF is a learnable monotonic transformation applied to each
input dimension independently:

    w(x) = 1 - (1 - x^a)^b

where a, b > 0 are learned per-dimension concentration parameters.

When a=b=1, this is the identity. The GP kernel operates on warped inputs,
effectively adapting to non-stationary response surfaces.

All 7 dims are continuous geometry/component parameters now. By default we
warp all dimensions, but can restrict to a subset if needed.

Reference:
    Snoek et al., "Input Warping for Bayesian Optimization of Non-Stationary
    Functions", ICML 2014.
"""

import torch
from botorch.models.transforms.input import Warp

from config import N_DIMS


def get_input_warp(dim: int = N_DIMS, indices: list[int] | None = None) -> Warp:
    """Create a Kumaraswamy input warping transform.

    Args:
        dim: number of input dimensions (7 for PA geo)
        indices: which dims to warp. If None, warps all dims.

    Returns:
        BoTorch Warp transform with learnable per-dimension parameters.
    """
    if indices is None:
        indices = list(range(dim))
    return Warp(d=dim, indices=indices)

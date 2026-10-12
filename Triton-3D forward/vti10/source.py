"""Define the two supported 3-D double-couple moment tensors.

Stress injection uses the negative tensor entries in the solver
component order [sxx, syy, szz, sxy, sxz, syz].
"""

from __future__ import annotations
import math
import numpy as np


def moment_tensor(kind: str, dtype=np.float64) -> np.ndarray:
    """Return either double-couple moment tensor from Simulation3_homo_downhole.m."""
    s = math.sqrt(0.5)
    k = kind.lower()
    if k == "dc1":
        m = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]], dtype=dtype) * s
    elif k == "dc2":
        m = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, -1.0, 0.0]], dtype=dtype) * s
    else:
        raise ValueError("source must be dc1 or dc2")
    return m


def stress_source_weights(kind: str, dtype=np.float64) -> np.ndarray:
    """Return single-node unsplit source weights in [sxx,syy,szz,sxy,sxz,syz] order."""
    m = moment_tensor(kind, dtype=dtype)
    return np.asarray([-m[0,0], -m[1,1], -m[2,2], -m[0,1], -m[0,2], -m[1,2]], dtype=dtype)

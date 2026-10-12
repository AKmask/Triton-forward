"""Generate FDwave3D-style staggered first-derivative coefficients.

The 3-D solver supports spatial orders 4, 6, 8, and 10; coefficient
tuples are padded to five entries for the shared Triton signatures.
"""

from __future__ import annotations
import numpy as np


def staggered_diff_coeff(order: int, dtype=np.float64) -> np.ndarray:
    """Return staggered-grid first-derivative coefficients matching FDwave3D DiffCoef(N,'s')."""
    if order not in (4, 6, 8, 10):
        raise ValueError("order must be 4, 6, 8, or 10")
    half = order // 2
    A = np.zeros((half, half), dtype=np.float64)
    b = np.zeros(half, dtype=np.float64)
    b[0] = 1.0
    # Match odd Taylor moments for the staggered first derivative.
    for i in range(half):
        p = 2 * (i + 1) - 1
        for j in range(half):
            A[i, j] = float(2 * (j + 1) - 1) ** p
    c = np.linalg.solve(A, b)
    return c.astype(dtype, copy=False)


def padded_coefficients(order: int):
    """Pad the selected stencil to the five coefficient slots of the kernels."""
    c = staggered_diff_coeff(order, dtype=np.float64)
    out = np.zeros(5, dtype=np.float64)
    out[: len(c)] = c
    return tuple(float(v) for v in out)

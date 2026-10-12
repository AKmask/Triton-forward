"""Generate staggered finite-difference coefficients and a 2-D CFL bound.

Supported spatial orders are 2, 4, 6, 8, and 10.
"""

from __future__ import annotations

import numpy as np


SUPPORTED_ORDERS = (2, 4, 6, 8, 10)


def staggered_diff_coeff(order: int, dtype=np.float64) -> np.ndarray:
    """Return centered staggered-grid first-derivative coefficients."""
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"order must be one of {SUPPORTED_ORDERS}, got {order}")
    half = order // 2
    matrix = np.zeros((half, half), dtype=np.float64)
    right_hand_side = np.zeros(half, dtype=np.float64)
    right_hand_side[0] = 1.0
    # Match odd Taylor moments to obtain the requested spatial accuracy.
    for row in range(half):
        power = 2 * (row + 1) - 1
        for column in range(half):
            matrix[row, column] = float(2 * (column + 1) - 1) ** power
    coefficients = np.linalg.solve(matrix, right_hand_side)
    return coefficients.astype(dtype, copy=False)


def padded_coefficients(order: int) -> tuple[float, float, float, float, float]:
    """Pad coefficients to the five compile-time slots used by Triton kernels."""
    coefficients = staggered_diff_coeff(order)
    padded = np.zeros(5, dtype=np.float64)
    padded[: coefficients.size] = coefficients
    return tuple(float(value) for value in padded)


def conservative_cfl_limit(dx: float, dz: float, vmax: float, order: int) -> float:
    """Return the conservative time-step bound from spacing, speed, and order."""
    coefficient_sum = float(np.sum(np.abs(staggered_diff_coeff(order))))
    return min(float(dx), float(dz)) / (np.sqrt(2.0) * coefficient_sum * float(vmax))

"""Partition the active 2-D stencil domain into interior and PML cells.

Nonzero damping in either direction identifies PML. The outer HALF cells
are excluded so all derivative neighbours remain inside the arrays.
"""

from __future__ import annotations

import numpy as np


def build_region_indices(case, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened interior and PML indices after removing the stencil halo."""
    z, x = np.indices(case.shape, dtype=np.int32)
    active = (x >= half) & (x < case.nx - half) & (z >= half) & (z < case.nz - half)
    pml_flag = (case.pmlx != 0.0) | (case.pmlz != 0.0)
    linear = np.arange(case.nx * case.nz, dtype=np.int64).reshape(case.shape)
    interior_indices = np.ascontiguousarray(linear[active & ~pml_flag].reshape(-1))
    pml_indices = np.ascontiguousarray(linear[active & pml_flag].reshape(-1))
    return interior_indices, pml_indices


def region_masks(case, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Return interior and PML masks for the NumPy reference updates."""
    z, x = np.indices(case.shape)
    active = (x >= half) & (x < case.nx - half) & (z >= half) & (z < case.nz - half)
    pml_flag = (case.pmlx != 0.0) | (case.pmlz != 0.0)
    return active & ~pml_flag, active & pml_flag


def region_stats(case, half: int) -> dict:
    """Summarize total, active, interior, and PML cell counts."""
    interior, pml = build_region_indices(case, half)
    active = int(interior.size + pml.size)
    return {
        "total_points": int(case.nx * case.nz),
        "fd_active_points": active,
        "interior_points": int(interior.size),
        "pml_points": int(pml.size),
        "pml_fraction_of_active": 0.0 if active == 0 else float(pml.size / active),
    }

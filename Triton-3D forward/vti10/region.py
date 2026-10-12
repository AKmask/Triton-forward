"""Build disjoint interior and PML index lists for the active 3-D stencil.

Classification uses the loaded damping arrays rather than PML thickness
alone; flattened indices follow the internal [z, y, x] layout.
"""

from __future__ import annotations
import numpy as np


def build_region_indices(case, half: int):
    """Partition the active stencil region into disjoint interior/PML indices using FDwave3D damping arrays."""
    nz, ny, nx = case.shape
    z, y, x = np.indices((nz, ny, nx), dtype=np.int32)
    fd_active = (
        (x >= half) & (x < nx-half) &
        (y >= half) & (y < ny-half) &
        (z >= half) & (z < nz-half)
    )
    # A point belongs to split-PML if any directional damping coefficient is nonzero.
    pml_flag = (case.pmlx != 0.0) | (case.pmly != 0.0) | (case.pmlz != 0.0)
    pml = fd_active & pml_flag
    interior = fd_active & (~pml_flag)
    linear = np.arange(nx*ny*nz, dtype=np.int64).reshape(nz, ny, nx)
    interior_idx = np.ascontiguousarray(linear[interior].reshape(-1))
    pml_idx = np.ascontiguousarray(linear[pml].reshape(-1))
    return interior_idx, pml_idx


def region_stats(case, half: int):
    """Report active interior/PML counts and the PML fraction."""
    i, p = build_region_indices(case, half)
    total = int(case.nx*case.ny*case.nz)
    active = int(i.size+p.size)
    return {
        "total_points": total,
        "fd_active_points": active,
        "interior_points": int(i.size),
        "pml_points": int(p.size),
        "pml_fraction_of_active": 0.0 if active == 0 else float(p.size/active),
    }

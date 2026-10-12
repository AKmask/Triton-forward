"""Validate 2-D VTI material fields and construct in-plane stiffnesses.

The validation permits the acoustic water-layer limit with zero shear
moduli and provides a conservative velocity bound for the CFL check.
"""

from __future__ import annotations

import numpy as np


INDEPENDENT_VTI_FIELDS = ("C11", "C13", "C33", "C44", "C66")


def _first_bad(mask: np.ndarray) -> tuple[int, int]:
    return tuple(int(value) for value in np.argwhere(mask)[0])


def _positive(name: str, array: np.ndarray) -> None:
    mask = array <= 0.0
    if np.any(mask):
        index = _first_bad(mask)
        raise ValueError(f"{name} must be positive; first bad [z,x]={index}, value={array[index]:.8e}")


def _nonnegative(name: str, array: np.ndarray) -> None:
    mask = array < 0.0
    if np.any(mask):
        index = _first_bad(mask)
        raise ValueError(f"{name} must be non-negative; first bad [z,x]={index}, value={array[index]:.8e}")


def validate_vti_material(
    rho: np.ndarray,
    c11: np.ndarray,
    c13: np.ndarray,
    c33: np.ndarray,
    c44: np.ndarray,
    c66: np.ndarray,
) -> dict:
    """Validate VTI stability, including acoustic cells, and report material ranges."""
    arrays = {
        "rho": np.asarray(rho, dtype=np.float64),
        "C11": np.asarray(c11, dtype=np.float64),
        "C13": np.asarray(c13, dtype=np.float64),
        "C33": np.asarray(c33, dtype=np.float64),
        "C44": np.asarray(c44, dtype=np.float64),
        "C66": np.asarray(c66, dtype=np.float64),
    }
    shape = arrays["rho"].shape
    if len(shape) != 2:
        raise ValueError(f"2-D material arrays must have [z,x] shape, got {shape}")
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{name}.shape={array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)):
            index = _first_bad(~np.isfinite(array))
            raise ValueError(f"{name} contains NaN or Inf at [z,x]={index}")
    _positive("rho", arrays["rho"])
    _positive("C33", arrays["C33"])
    # Hess contains an acoustic water layer.  C44=C66=0 and a zero coupled
    # stability margin are the valid positive-semidefinite acoustic limit.
    _nonnegative("C44", arrays["C44"])
    _nonnegative("C66", arrays["C66"])
    horizontal_bulk = arrays["C11"] - arrays["C66"]
    _positive("C11-C66", horizontal_bulk)
    coupled_scale = horizontal_bulk * arrays["C33"]
    coupled_margin = coupled_scale - arrays["C13"] ** 2
    tolerance = 1.0e-10 * np.maximum(coupled_scale, 1.0)
    unstable = coupled_margin < -tolerance
    if np.any(unstable):
        index = _first_bad(unstable)
        raise ValueError(
            "(C11-C66)*C33-C13^2 must be non-negative (within roundoff); "
            f"first bad [z,x]={index}, value={coupled_margin[index]:.8e}"
        )
    normalized_margin = coupled_margin / coupled_scale
    ranges = {
        name: {"min": float(np.min(array)), "max": float(np.max(array))}
        for name, array in arrays.items()
    }
    # A conservative upper bound for the in-plane Christoffel eigenvalue.
    stiffness_bound = np.maximum.reduce((arrays["C11"], arrays["C33"], arrays["C44"]))
    stiffness_bound = stiffness_bound + 0.5 * np.abs(arrays["C13"] + arrays["C44"])
    vmax_bound = float(np.sqrt(np.max(stiffness_bound / arrays["rho"])))
    return {
        "shape_zx": [int(value) for value in shape],
        "independent_coefficients": list(INDEPENDENT_VTI_FIELDS),
        "ranges": ranges,
        "horizontal_bulk_margin_min": float(np.min(horizontal_bulk)),
        "coupled_stability_margin_min": float(np.min(coupled_margin)),
        "normalized_stability_margin_min": float(np.min(normalized_margin)),
        "acoustic_cell_count": int(np.count_nonzero((arrays["C44"] == 0.0) & (arrays["C66"] == 0.0))),
        "allows_acoustic_semidefinite_limit": True,
        "conservative_vmax_m_s": vmax_bound,
    }


def construct_inplane_stiffness(c11, c13, c33, c44) -> dict[str, np.ndarray]:
    """Map C44 to the C55 shear field used by the x-z propagation equations."""
    return {
        "C11": np.ascontiguousarray(c11, dtype=np.float64),
        "C13": np.ascontiguousarray(c13, dtype=np.float64),
        "C33": np.ascontiguousarray(c33, dtype=np.float64),
        "C55": np.ascontiguousarray(c44, dtype=np.float64),
    }

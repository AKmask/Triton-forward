"""Expand and validate the five independent 3-D VTI stiffness coefficients.

The symmetry relations are C12=C11-2*C66, C22=C11, C23=C13, and C55=C44.
Density and stiffnesses are interpreted in SI units.
"""

from __future__ import annotations

import numpy as np


INDEPENDENT_VTI_COEFFICIENTS = ("C11", "C13", "C33", "C44", "C66")
REDUNDANT_VTI_COEFFICIENTS = ("C12", "C22", "C23", "C55")


def construct_vti_stiffness(c11, c13, c33, c44, c66) -> dict[str, np.ndarray]:
    """Expand the five independent VTI coefficients to the nine solver fields."""
    c11 = np.ascontiguousarray(c11, dtype=np.float64)
    c13 = np.ascontiguousarray(c13, dtype=np.float64)
    c33 = np.ascontiguousarray(c33, dtype=np.float64)
    c44 = np.ascontiguousarray(c44, dtype=np.float64)
    c66 = np.ascontiguousarray(c66, dtype=np.float64)
    return {
        "C11": c11,
        "C12": np.ascontiguousarray(c11 - 2.0 * c66),
        "C13": c13,
        "C22": c11.copy(),
        "C23": c13.copy(),
        "C33": c33,
        "C44": c44,
        "C55": c44.copy(),
        "C66": c66,
    }


def _first_bad(array: np.ndarray, predicate) -> tuple[int, ...]:
    flat_index = int(np.flatnonzero(predicate(array))[0])
    return tuple(int(value) for value in np.unravel_index(flat_index, array.shape))


def _require_positive(name: str, array: np.ndarray) -> None:
    if np.any(array <= 0.0):
        index = _first_bad(array, lambda value: value <= 0.0)
        raise ValueError(f"{name} must be positive everywhere; first invalid internal [z,y,x]={index}, value={array[index]:.8e}")


def validate_vti_material(
    rho: np.ndarray,
    c11: np.ndarray,
    c13: np.ndarray,
    c33: np.ndarray,
    c44: np.ndarray,
    c66: np.ndarray,
) -> dict:
    """Validate density and the positive-definiteness conditions of a VTI tensor."""
    arrays = {
        "rho": np.asarray(rho, dtype=np.float64),
        "C11": np.asarray(c11, dtype=np.float64),
        "C13": np.asarray(c13, dtype=np.float64),
        "C33": np.asarray(c33, dtype=np.float64),
        "C44": np.asarray(c44, dtype=np.float64),
        "C66": np.asarray(c66, dtype=np.float64),
    }
    shape = arrays["rho"].shape
    if len(shape) != 3:
        raise ValueError(f"VTI material arrays must be three-dimensional, got rho.shape={shape}")
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{name}.shape={array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)):
            index = _first_bad(array, lambda value: ~np.isfinite(value))
            raise ValueError(f"{name} contains NaN or Inf; first invalid internal [z,y,x]={index}")

    _require_positive("rho", arrays["rho"])
    _require_positive("C33", arrays["C33"])
    _require_positive("C44", arrays["C44"])
    _require_positive("C66", arrays["C66"])

    horizontal_bulk = arrays["C11"] - arrays["C66"]
    _require_positive("C11-C66", horizontal_bulk)
    coupled_margin = horizontal_bulk * arrays["C33"] - arrays["C13"] ** 2
    _require_positive("(C11-C66)*C33-C13^2", coupled_margin)
    normalized_margin = coupled_margin / (horizontal_bulk * arrays["C33"])

    ranges = {
        name: {"min": float(np.min(array)), "max": float(np.max(array))}
        for name, array in arrays.items()
    }
    return {
        "shape_zyx": [int(value) for value in shape],
        "independent_coefficients": list(INDEPENDENT_VTI_COEFFICIENTS),
        "ranges": ranges,
        "horizontal_bulk_margin_min": float(np.min(horizontal_bulk)),
        "coupled_stability_margin_min": float(np.min(coupled_margin)),
        "normalized_stability_margin_min": float(np.min(normalized_margin)),
    }


def validate_redundant_vti_coefficients(
    supplied: dict[str, np.ndarray], expected: dict[str, np.ndarray], rtol: float = 1e-10
) -> dict:
    """Check optional redundant fields without making them authoritative."""
    checked = []
    max_relative_error = 0.0
    for name in REDUNDANT_VTI_COEFFICIENTS:
        if name not in supplied:
            continue
        actual = np.asarray(supplied[name], dtype=np.float64)
        reference = np.asarray(expected[name], dtype=np.float64)
        if actual.shape != reference.shape:
            raise ValueError(f"Redundant {name}.shape={actual.shape}, expected {reference.shape}")
        if not np.all(np.isfinite(actual)):
            raise ValueError(f"Redundant {name} contains NaN or Inf")
        scale = max(float(np.max(np.abs(reference))), 1.0)
        relative_error = float(np.max(np.abs(actual - reference))) / scale
        if relative_error > rtol:
            raise ValueError(
                f"Input {name} violates VTI symmetry: maximum relative error={relative_error:.8e}, tolerance={rtol:.8e}"
            )
        checked.append(name)
        max_relative_error = max(max_relative_error, relative_error)
    return {"checked_fields": checked, "max_relative_error": max_relative_error, "rtol": float(rtol)}

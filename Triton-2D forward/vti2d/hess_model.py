"""Read Hess MAT arrays, convert units, and apply spatial transformations.

The loader accepts stiffness or Thomsen inputs and returns contiguous
[z, x] material fields plus velocity and anisotropy diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


DEFAULT_FILES = {
    "rho": {"filename": "Hess_rho.mat", "variable": "rho"},
    "C11": {"filename": "Hess_C11.mat", "variable": "c11"},
    "C13": {"filename": "Hess_C13.mat", "variable": "c13"},
    "C33": {"filename": "Hess_C33.mat", "variable": "c33"},
    "C44": {"filename": "Hess_C44.mat", "variable": "c44"},
    "C66": None,
    "vp": {"filename": "Hess_vp.mat", "variable": "vp"},
    "vs": {"filename": "Hess_vs.mat", "variable": "vs"},
    "epsilon": {"filename": "Hess_epsilon.mat", "variable": "epsilon"},
    "delta": {"filename": "Hess_delta.mat", "variable": "delta"},
}


def _resolve_directory(value: str, base_dir: Path) -> Path:
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = base_dir / directory
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Hess model directory does not exist: {directory}")
    return directory


def _field_spec(files: dict, name: str, required: bool = True) -> dict | None:
    raw = files.get(name, DEFAULT_FILES[name])
    if raw is None:
        if required:
            raise ValueError(f"hess_model.files.{name} is required")
        return None
    if isinstance(raw, str):
        default = DEFAULT_FILES[name]
        variable = name if default is None else default["variable"]
        return {"filename": raw, "variable": variable}
    if not isinstance(raw, dict):
        raise ValueError(f"hess_model.files.{name} must be a filename, an object, or null")
    filename = str(raw.get("filename", "")).strip()
    variable = str(raw.get("variable", "")).strip()
    if not filename or not variable:
        raise ValueError(f"hess_model.files.{name} requires filename and variable")
    return {"filename": filename, "variable": variable}


def _load_mat_array(path: Path, expected_variable: str) -> np.ndarray:
    """Read a real 2-D field, allowing one unambiguous fallback MAT variable."""
    if not path.is_file():
        raise FileNotFoundError(f"Hess model file does not exist: {path}")
    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"MATLAB v7.3 is not supported for {path}; save the array as MATLAB v7 MAT"
        ) from exc
    if expected_variable in payload:
        array = payload[expected_variable]
    else:
        candidates = {
            key: value
            for key, value in payload.items()
            if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2
        }
        if len(candidates) != 1:
            names = ", ".join(sorted(candidates)) or "<none>"
            raise KeyError(
                f"{path.name} has no variable {expected_variable!r}; "
                f"2-D candidates: {names}"
            )
        array = next(iter(candidates.values()))
    array = np.asarray(array)
    if array.ndim != 2 or np.iscomplexobj(array):
        raise ValueError(f"{path.name}:{expected_variable} must be a real 2-D array")
    if not np.all(np.isfinite(array)):
        bad = tuple(int(index) for index in np.argwhere(~np.isfinite(array))[0])
        raise ValueError(f"{path.name}:{expected_variable} contains NaN/Inf at [z,x]={bad}")
    return np.ascontiguousarray(array, dtype=np.float64)


def _positive_integer(parent: dict, name: str, default: int) -> int:
    raw = parent.get(name, default)
    value = int(raw)
    if value < 1 or float(raw) != value:
        raise ValueError(f"hess_model.stride.{name} must be a positive integer")
    return value


def _crop_bound(parent: dict, name: str, default: int, upper: int) -> int:
    raw = parent.get(name)
    value = default if raw is None else int(raw)
    if not 0 <= value <= upper:
        raise ValueError(f"hess_model.crop.{name}={value} is outside [0,{upper}]")
    return value


def _transform_arrays(arrays: dict[str, np.ndarray], config: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Apply transpose, flips, crop, and spatial decimation to every field."""
    shape = next(iter(arrays.values())).shape
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"Hess {name}.shape={array.shape}, expected common shape {shape}")
    if bool(config.get("transpose", False)):
        arrays = {name: np.ascontiguousarray(array.T) for name, array in arrays.items()}
        shape = shape[::-1]
    if bool(config.get("flip_z", False)):
        arrays = {name: np.ascontiguousarray(array[::-1, :]) for name, array in arrays.items()}
    if bool(config.get("flip_x", False)):
        arrays = {name: np.ascontiguousarray(array[:, ::-1]) for name, array in arrays.items()}

    crop = config.get("crop", {})
    if not isinstance(crop, dict):
        raise ValueError("hess_model.crop must be an object")
    nz, nx = shape
    z_start = _crop_bound(crop, "z_start", 0, nz)
    z_end = _crop_bound(crop, "z_end", nz, nz)
    x_start = _crop_bound(crop, "x_start", 0, nx)
    x_end = _crop_bound(crop, "x_end", nx, nx)
    if z_end <= z_start or x_end <= x_start:
        raise ValueError("Hess crop must satisfy z_start < z_end and x_start < x_end")
    stride = config.get("stride", {})
    if not isinstance(stride, dict):
        raise ValueError("hess_model.stride must be an object")
    stride_z = _positive_integer(stride, "z", 1)
    stride_x = _positive_integer(stride, "x", 1)
    transformed = {
        name: np.ascontiguousarray(array[z_start:z_end:stride_z, x_start:x_end:stride_x])
        for name, array in arrays.items()
    }
    info = {
        "raw_shape_zx": [int(shape[0]), int(shape[1])],
        "crop_zx": [z_start, z_end, x_start, x_end],
        "stride_zx": [stride_z, stride_x],
        "shape_zx": [int(next(iter(transformed.values())).shape[0]), int(next(iter(transformed.values())).shape[1])],
        "transpose": bool(config.get("transpose", False)),
        "flip_z": bool(config.get("flip_z", False)),
        "flip_x": bool(config.get("flip_x", False)),
    }
    return transformed, info


def _derived_diagnostics(material: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derive vertical velocities and Thomsen parameters for model inspection."""
    rho = material["rho"]
    c11, c13, c33, c44 = (material[name] for name in ("C11", "C13", "C33", "C44"))
    vp = np.sqrt(np.maximum(c33, 0.0) / rho)
    vs = np.sqrt(np.maximum(c44, 0.0) / rho)
    epsilon = 0.5 * (c11 / c33 - 1.0)
    denominator = 2.0 * c33 * (c33 - c44)
    numerator = (c13 + c44) ** 2 - (c33 - c44) ** 2
    delta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0.0)
    return {
        "vp_m_s": np.ascontiguousarray(vp),
        "vs_m_s": np.ascontiguousarray(vs),
        "epsilon": np.ascontiguousarray(epsilon),
        "delta": np.ascontiguousarray(delta),
    }


def load_hess_material(config: dict, base_dir: str | Path) -> dict[str, Any]:
    """Load Hess VTI arrays and return SI-valued fields in [z,x] order.

    ``parameterization=stiffness`` reads rho/C11/C13/C33/C44 directly.
    ``parameterization=thomsen`` reads rho/vp/epsilon/delta and either C44
    or vs, then constructs the in-plane VTI stiffnesses.
    """
    if not isinstance(config, dict):
        raise ValueError("hess_model must be an object")
    if not str(config.get("directory", "")).strip():
        raise ValueError("hess_model.directory is required")
    directory = _resolve_directory(str(config["directory"]), Path(base_dir).resolve())
    parameterization = str(config.get("parameterization", "stiffness")).lower()
    if parameterization not in {"stiffness", "thomsen"}:
        raise ValueError("hess_model.parameterization must be stiffness or thomsen")
    files = config.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("hess_model.files must be an object")
    scales = config.get("scales", {})
    if not isinstance(scales, dict):
        raise ValueError("hess_model.scales must be an object")
    rho_scale = float(scales.get("density", 1.0))
    stiffness_scale = float(scales.get("stiffness", 1.0))
    velocity_scale = float(scales.get("velocity", 0.3048))
    if not all(np.isfinite(value) and value > 0.0 for value in (rho_scale, stiffness_scale, velocity_scale)):
        raise ValueError("Hess density/stiffness/velocity scales must be finite and positive")

    raw: dict[str, np.ndarray] = {}
    rho_spec = _field_spec(files, "rho")
    raw["rho"] = _load_mat_array(directory / rho_spec["filename"], rho_spec["variable"]) * rho_scale
    c66_source = "file"
    if parameterization == "stiffness":
        for name in ("C11", "C13", "C33", "C44"):
            spec = _field_spec(files, name)
            raw[name] = _load_mat_array(directory / spec["filename"], spec["variable"]) * stiffness_scale
    else:
        for name in ("vp", "epsilon", "delta"):
            spec = _field_spec(files, name)
            raw[name] = _load_mat_array(directory / spec["filename"], spec["variable"])
        raw["vp"] *= velocity_scale
        shear_mode = str(config.get("thomsen_shear", "C44")).lower()
        if shear_mode == "c44":
            spec = _field_spec(files, "C44")
            raw["C44"] = _load_mat_array(directory / spec["filename"], spec["variable"]) * stiffness_scale
        elif shear_mode == "vs":
            spec = _field_spec(files, "vs")
            raw["vs"] = _load_mat_array(directory / spec["filename"], spec["variable"]) * velocity_scale
        else:
            raise ValueError("hess_model.thomsen_shear must be C44 or vs")

    c66_spec = _field_spec(files, "C66", required=False)
    if c66_spec is not None:
        raw["C66"] = _load_mat_array(directory / c66_spec["filename"], c66_spec["variable"]) * stiffness_scale
    else:
        c66_source = "C44 (2-D qP-qSV compatibility field; not used by propagation)"

    raw, transform = _transform_arrays(raw, config)
    if parameterization == "thomsen":
        rho, vp = raw["rho"], raw.pop("vp")
        epsilon, delta = raw.pop("epsilon"), raw.pop("delta")
        if "vs" in raw:
            vs = raw.pop("vs")
            raw["C44"] = rho * vs**2
        c33, c44 = rho * vp**2, raw["C44"]
        radicand = (c33 - c44) ** 2 + 2.0 * delta * c33 * (c33 - c44)
        tolerance = 1.0e-12 * np.maximum((c33 - c44) ** 2, 1.0)
        if np.any(radicand < -tolerance):
            bad = tuple(int(index) for index in np.argwhere(radicand < -tolerance)[0])
            raise ValueError(f"Thomsen conversion has a negative C13 radicand at [z,x]={bad}")
        raw["C33"] = c33
        raw["C11"] = c33 * (1.0 + 2.0 * epsilon)
        raw["C13"] = np.sqrt(np.maximum(radicand, 0.0)) - c44
    if "C66" not in raw:
        raw["C66"] = raw["C44"].copy()

    storage_dtype = str(config.get("storage_dtype", "float32")).lower()
    if storage_dtype not in {"float32", "float64"}:
        raise ValueError("hess_model.storage_dtype must be float32 or float64")
    dtype = np.float32 if storage_dtype == "float32" else np.float64
    material = {
        name: np.ascontiguousarray(raw[name], dtype=dtype)
        for name in ("rho", "C11", "C13", "C33", "C44", "C66")
    }
    diagnostics = {
        name: np.ascontiguousarray(array, dtype=dtype)
        for name, array in _derived_diagnostics(material).items()
    }
    return {
        "directory": directory,
        "parameterization": parameterization,
        "storage_dtype": storage_dtype,
        "material": material,
        "diagnostics": diagnostics,
        "transform": transform,
        "scales": {
            "density": rho_scale,
            "stiffness": stiffness_scale,
            "velocity": velocity_scale,
        },
        "c66_source": c66_source,
    }

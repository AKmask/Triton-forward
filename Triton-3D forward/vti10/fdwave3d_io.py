"""Read FDwave3D-compatible MAT inputs and prepare a 3-D VTI case.

Material arrays change from MATLAB [z, x, y] to contiguous [z, y, x].
Geometry becomes zero-based, and staggered parameters are recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

from .vti_stiffness import (
    REDUNDANT_VTI_COEFFICIENTS,
    construct_vti_stiffness,
    validate_redundant_vti_coefficients,
    validate_vti_material,
)


def _mat_path(case_dir: Path, stem: str) -> Optional[Path]:
    for path in (case_dir / stem, case_dir / f"{stem}.mat"):
        if path.exists():
            return path
    return None


def _load_mat(path: Path) -> dict:
    try:
        return loadmat(path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(f"{path} is a MATLAB v7.3 file; save input files in -v7 format.") from exc


def _find_key(mat: dict, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in mat:
            return alias
    lower_to_key = {str(key).lower(): key for key in mat}
    for alias in aliases:
        key = lower_to_key.get(alias.lower())
        if key is not None:
            return key
    return None


def _scalar(mat: dict, name: str, typ=float):
    key = _find_key(mat, (name,))
    if key is None:
        raise KeyError(f"Missing variable {name}")
    return typ(np.asarray(mat[key]).squeeze())


def _zyx(array, dtype=np.float64):
    array = np.asarray(array, dtype=dtype)
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional array, got shape={array.shape}")
    # FDwave3D MATLAB model [z,x,y] -> Python/Triton internal [z,y,x]
    return np.ascontiguousarray(np.transpose(array, (0, 2, 1)))


def _model_array(mat: dict, aliases: tuple[str, ...], display_name: str) -> np.ndarray:
    key = _find_key(mat, aliases)
    if key is None:
        raise KeyError(f"model.mat is missing VTI input {display_name} (accepted names: {', '.join(aliases)})")
    return _zyx(mat[key])


def _optional_model_array(mat: dict, name: str) -> np.ndarray | None:
    key = _find_key(mat, (name, name.lower()))
    return None if key is None else _zyx(mat[key])


def _vec0(mat: dict, name: str) -> np.ndarray:
    """Read a MATLAB coordinate vector and convert it to zero-based indices."""
    key = _find_key(mat, (name,))
    if key is None:
        raise KeyError(name)
    return np.asarray(mat[key]).squeeze().astype(np.int64).reshape(-1) - 1


def _matstr(mat: dict, name: str, default: str) -> str:
    key = _find_key(mat, (name,))
    if key is None:
        return default
    value = np.asarray(mat[key]).squeeze()
    try:
        value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if isinstance(value, np.ndarray):
        return "".join(str(item) for item in value.reshape(-1)).strip()
    return str(value).strip()


def _unwrap_mat_struct(obj):
    while isinstance(obj, np.ndarray) and obj.size == 1:
        obj = obj.item()
    return obj


def _struct_field(obj, name: str):
    obj = _unwrap_mat_struct(obj)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise KeyError(name)


@dataclass
class FDWave3DCase:
    """3-D case data with internal [z, y, x] fields and zero-based geometry."""
    case_dir: Path
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    c11: np.ndarray
    c12: np.ndarray
    c13: np.ndarray
    c22: np.ndarray
    c23: np.ndarray
    c33: np.ndarray
    c44: np.ndarray
    c55: np.ndarray
    c66: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    c44s: np.ndarray
    c55s: np.ndarray
    c66s: np.ndarray
    pmlx: np.ndarray
    pmly: np.ndarray
    pmlz: np.ndarray
    dt: float
    total_time: float
    nt: int
    source: np.ndarray
    src_x: int
    src_y: int
    src_z: int
    rec_x: np.ndarray
    rec_y: np.ndarray
    rec_z: np.ndarray
    nab: int
    bc_type: str
    bc_name: str
    material_model: str
    stiffness_source: str
    vti_validation: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nz, self.ny, self.nx

    @property
    def nrec(self) -> int:
        return int(self.rec_x.size)


def ricker(total_time=0.24, dt=0.0003, f0=60.0, t0=0.03):
    """Return a unit-amplitude Ricker wavelet for the fallback source."""
    count = int(round(total_time / dt)) + 1
    time = dt * np.arange(count, dtype=np.float64)
    tau = time - t0
    source = (1.0 - 2.0 * (np.pi * f0 * tau) ** 2) * np.exp(-(np.pi * f0 * tau) ** 2)
    return np.ascontiguousarray(source, dtype=np.float64)


def _pad_full(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    out = np.empty(shape, dtype=np.float64)
    out.fill(float(np.asarray(array).reshape(-1)[-1]))
    z_size, y_size, x_size = array.shape
    out[:z_size, :y_size, :x_size] = array
    return np.ascontiguousarray(out)


def _derive(shape, rho, c44, c55, c66):
    """Recompute staggered inverse density and shear moduli from authoritative VTI fields."""
    inverse_density = 1.0 / rho
    bx0 = 0.5 * (inverse_density[:, :, :-1] + inverse_density[:, :, 1:])
    by0 = 0.5 * (inverse_density[:, :-1, :] + inverse_density[:, 1:, :])
    bz0 = 0.5 * (inverse_density[:-1, :, :] + inverse_density[1:, :, :])

    def average_eight(array):
        return 0.125 * (
            array[1:-1, 1:-1, 1:-1]
            + array[2:, 1:-1, 1:-1]
            + array[1:-1, 1:-1, 2:]
            + array[1:-1, 2:, 1:-1]
            + array[2:, 1:-1, 2:]
            + array[2:, 2:, 1:-1]
            + array[1:-1, 2:, 2:]
            + array[2:, 2:, 2:]
        )

    return (
        _pad_full(bx0, shape),
        _pad_full(by0, shape),
        _pad_full(bz0, shape),
        _pad_full(average_eight(c44), shape),
        _pad_full(average_eight(c55), shape),
        _pad_full(average_eight(c66), shape),
    )


def _load_pml_arrays(bp: Path | None, shape):
    """Load and validate PML damping, or return zeros when PML data are absent."""
    if bp is None:
        zeros = np.zeros(shape, dtype=np.float64)
        return zeros.copy(), zeros.copy(), zeros.copy(), "topABC", "NONE", 0
    bc = _load_mat(bp)
    if _find_key(bc, ("nAB",)) is not None:
        nab = _scalar(bc, "nAB", int)
    elif _find_key(bc, ("NAB",)) is not None:
        nab = _scalar(bc, "NAB", int)
    else:
        nab = 0
    bc_type = _matstr(bc, "BCtype", "topABC")
    bc_name = _matstr(bc, "BCname", "PML")
    if "BC" not in bc or bc_name.upper() != "PML":
        zeros = np.zeros(shape, dtype=np.float64)
        return zeros.copy(), zeros.copy(), zeros.copy(), bc_type, bc_name, nab
    obj = bc["BC"]
    try:
        px = _zyx(_struct_field(obj, "pmlx"))
        py = _zyx(_struct_field(obj, "pmly"))
        pz = _zyx(_struct_field(obj, "pmlz"))
    except KeyError as exc:
        raise KeyError(f"The BC structure is missing a split-PML field: {exc}") from exc
    for name, array in (("pmlx", px), ("pmly", py), ("pmlz", pz)):
        if array.shape != shape:
            raise ValueError(f"{name}.shape={array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must contain finite, non-negative damping coefficients")
    return px, py, pz, bc_type, bc_name, nab


def _validate_coordinates(name, x, y, z, shape):
    """Check coordinate-vector lengths and zero-based model bounds."""
    if not (x.size == y.size == z.size):
        raise ValueError(f"{name} has mismatched x/y/z coordinate counts: {x.size}/{y.size}/{z.size}")
    nz, ny, nx = shape
    valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & (z >= 0) & (z < nz)
    if not np.all(valid):
        index = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            f"{name} coordinates are out of bounds: entry {index} has internal [x,y,z]=[{x[index]},{y[index]},{z[index]}], "
            f"grid [nx,ny,nz]=[{nx},{ny},{nz}]"
        )


def load_fdwave3d_case(case_dir: str | Path, nab_fallback=20) -> FDWave3DCase:
    """Load a five-coefficient VTI case while preserving the 09 solver conventions."""
    case_dir = Path(case_dir).expanduser().resolve()
    model_path = _mat_path(case_dir, "model")
    if model_path is None:
        raise FileNotFoundError(f"Neither model nor model.mat was found in {case_dir}")
    model = _load_mat(model_path)

    nx, ny, nz = _scalar(model, "nx", int), _scalar(model, "ny", int), _scalar(model, "nz", int)
    dx, dy, dz = _scalar(model, "dx"), _scalar(model, "dy"), _scalar(model, "dz")
    if min(nx, ny, nz) < 3 or min(dx, dy, dz) <= 0.0:
        raise ValueError(f"Invalid grid parameters: n=({nx},{ny},{nz}), d=({dx},{dy},{dz})")
    shape = (nz, ny, nx)

    rho = _model_array(model, ("rhom", "rho"), "rho")
    c11 = _model_array(model, ("C11",), "C11")
    c13 = _model_array(model, ("C13",), "C13")
    c33 = _model_array(model, ("C33",), "C33")
    c44 = _model_array(model, ("C44",), "C44")
    c66 = _model_array(model, ("C66",), "C66")
    if rho.shape != shape:
        raise ValueError(f"rho.shape={rho.shape}, expected {shape} from nx/ny/nz")

    validation = validate_vti_material(rho, c11, c13, c33, c44, c66)
    full = construct_vti_stiffness(c11, c13, c33, c44, c66)
    redundant = {}
    for name in REDUNDANT_VTI_COEFFICIENTS:
        supplied = _optional_model_array(model, name)
        if supplied is not None:
            redundant[name] = supplied
    validation["redundant_coefficients"] = validate_redundant_vti_coefficients(
        redundant, full, rtol=1.0e-8
    )

    # VTI vertical phase velocities, retained as diagnostics only. The solver
    # consumes the stiffness tensor directly and does not reconstruct it from vp/vs.
    vp = np.ascontiguousarray(np.sqrt(c33 / rho))
    vs = np.ascontiguousarray(np.sqrt(c44 / rho))
    c12, c22, c23, c55 = full["C12"], full["C22"], full["C23"], full["C55"]

    bc_path = _mat_path(case_dir, "BC")
    pmlx, pmly, pmlz, bc_type, bc_name, nab_loaded = _load_pml_arrays(bc_path, shape)
    nab = nab_loaded if bc_path is not None else int(nab_fallback)

    source_path = _mat_path(case_dir, "source")
    if source_path is not None:
        source_model = _load_mat(source_path)
        dt = _scalar(source_model, "dt")
        total_time = _scalar(source_model, "T")
        nt = _scalar(source_model, "N", int)
        source_key = _find_key(source_model, ("src",))
        if source_key is None:
            raise KeyError("source.mat is missing variable src")
        source = np.asarray(source_model[source_key], dtype=np.float64).squeeze().reshape(-1)
        if source.size < nt:
            source = np.pad(source, (0, nt - source.size))
        source = np.ascontiguousarray(source[:nt])
    else:
        dt, total_time = 0.0003, 0.24
        source = ricker(total_time, dt)
        nt = len(source)
    if dt <= 0.0 or total_time < 0.0 or nt < 2 or not np.all(np.isfinite(source)):
        raise ValueError(f"Invalid source time parameters: dt={dt}, T={total_time}, N={nt}")

    geometry_source_path = _mat_path(case_dir, "geometry_src")
    if geometry_source_path is not None:
        geometry_source = _load_mat(geometry_source_path)
        try:
            sx = int(_vec0(geometry_source, "snx_vec")[0])
            sy = int(_vec0(geometry_source, "sny_vec")[0])
            sz = int(_vec0(geometry_source, "snz_vec")[0])
        except KeyError:
            sx, sy, sz = round(nx / 2) - 1, round(ny / 2) - 1, nz - nab - 1
    else:
        sx, sy, sz = round(nx / 2) - 1, round(ny / 2) - 1, nz - nab - 1
    _validate_coordinates("source", np.asarray([sx]), np.asarray([sy]), np.asarray([sz]), shape)

    geometry_receiver_path = _mat_path(case_dir, "geometry_rec")
    if geometry_receiver_path is not None:
        geometry_receiver = _load_mat(geometry_receiver_path)
        rx = _vec0(geometry_receiver, "rnx_vec")
        ry = _vec0(geometry_receiver, "rny_vec")
        rz = _vec0(geometry_receiver, "rnz_vec")
    else:
        rz = np.arange(nab, nz - nab, dtype=np.int64)
        rx = np.full_like(rz, nab)
        ry = np.full_like(rz, nab)
    _validate_coordinates("receiver", rx, ry, rz, shape)

    # Ignore possibly stale derived_param material fields. Recompute everything
    # from the density and five authoritative stiffness arrays on every load.
    bx, by, bz, c44s, c55s, c66s = _derive(shape, rho, c44, c55, c66)

    return FDWave3DCase(
        case_dir, nx, ny, nz, dx, dy, dz, vp, vs, rho,
        c11, c12, c13, c22, c23, c33, c44, c55, c66,
        bx, by, bz, c44s, c55s, c66s, pmlx, pmly, pmlz,
        dt, total_time, nt, source, sx, sy, sz,
        np.ascontiguousarray(rx), np.ascontiguousarray(ry), np.ascontiguousarray(rz),
        nab, bc_type, bc_name, "VTI", str(model_path), validation,
    )


# Explicit new name while preserving the entry point used by 09-derived scripts.
load_vti_case = load_fdwave3d_case

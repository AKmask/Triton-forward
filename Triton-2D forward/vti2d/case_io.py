"""Load and validate the MAT files of a 2-D Hess VTI case.

Arrays use [z, x] order. MATLAB geometry is converted from one-based
indices to zero-based Python indices before propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .fd_coeff import conservative_cfl_limit
from .source import SOURCE_KINDS
from .vti_stiffness import validate_vti_material


def _load(path: Path) -> dict:
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(f"MATLAB v7.3 is not supported for {path}; save it as v7 MAT") from exc


def _scalar(data: dict, name: str, cast=float):
    if name not in data:
        raise KeyError(f"missing {name}")
    return cast(np.asarray(data[name]).reshape(-1)[0])


def _optional_scalar(data: dict, name: str, cast=float):
    if name not in data:
        return None
    array = np.asarray(data[name]).reshape(-1)
    if array.size == 0:
        return None
    return cast(array[0])


def _text(value, default="") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        return "".join(array.reshape(-1).astype(str)).strip()
    return str(array.reshape(-1)[0]).strip()


def _array(data: dict, aliases: tuple[str, ...], shape: tuple[int, int], name: str) -> np.ndarray:
    key = next((alias for alias in aliases if alias in data), None)
    if key is None:
        raise KeyError(f"model.mat is missing {name}")
    array = np.asarray(data[key], dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name}.shape={array.shape}, expected [z,x]={shape}")
    return np.ascontiguousarray(array)


def _pad_full(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.empty(shape, dtype=np.float64)
    result.fill(float(array.reshape(-1)[-1]))
    result[: array.shape[0], : array.shape[1]] = array
    return np.ascontiguousarray(result)


def derive_staggered(rho: np.ndarray, c55: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average inverse density and shear stiffness onto the staggered grid."""
    inverse_density = 1.0 / rho
    bx = 0.5 * (inverse_density[:, :-1] + inverse_density[:, 1:])
    bz = 0.5 * (inverse_density[:-1, :] + inverse_density[1:, :])
    # Preserve version 10's staggered-shear placement convention: the reduced
    # corner array starts from source index 1 and is padded back to full shape.
    c55_corner = 0.25 * (
        c55[1:-1, 1:-1] + c55[2:, 1:-1] + c55[1:-1, 2:] + c55[2:, 2:]
    )
    return _pad_full(bx, rho.shape), _pad_full(bz, rho.shape), _pad_full(c55_corner, rho.shape)


def _struct_field(value, name: str):
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, dict):
        return value[name]
    raise KeyError(name)


@dataclass
class VTI2DCase:
    """Validated case data with [z, x] arrays and zero-based geometry."""
    case_dir: Path
    nx: int
    nz: int
    dx: float
    dz: float
    rho: np.ndarray
    c11: np.ndarray
    c13: np.ndarray
    c33: np.ndarray
    c44: np.ndarray
    c66: np.ndarray
    bx: np.ndarray
    bz: np.ndarray
    c55s: np.ndarray
    pmlx: np.ndarray
    pmlz: np.ndarray
    dt: float
    total_time: float
    nt: int
    source: np.ndarray
    source_name: str
    source_f0_hz: float | None
    source_t0_s: float | None
    source_scale: float | None
    default_source_kind: str
    src_x: int
    src_z: int
    rec_x: np.ndarray
    rec_z: np.ndarray
    nab: int
    bc_type: str
    pml_name: str
    pml_reflection_coefficient: float | None
    pml_polynomial_power: float | None
    validation: dict

    @property
    def shape(self) -> tuple[int, int]:
        return self.nz, self.nx

    @property
    def nrec(self) -> int:
        return int(self.rec_x.size)


def load_vti2d_case(case_dir: str | Path, order: int = 10) -> VTI2DCase:
    """Load case files and validate material, sampling, boundaries, and geometry."""
    directory = Path(case_dir).expanduser().resolve()
    required = ("model.mat", "source.mat", "BC.mat", "geometry_src.mat", "geometry_rec.mat")
    for filename in required:
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"missing {directory / filename}")
    model = _load(directory / "model.mat")
    nx, nz = _scalar(model, "nx", int), _scalar(model, "nz", int)
    dx, dz = _scalar(model, "dx"), _scalar(model, "dz")
    if min(nx, nz) <= order or min(dx, dz) <= 0.0:
        raise ValueError(f"invalid grid/order: nx={nx}, nz={nz}, dx={dx}, dz={dz}, order={order}")
    shape = (nz, nx)
    rho = _array(model, ("rhom", "rho"), shape, "rho")
    c11 = _array(model, ("C11",), shape, "C11")
    c13 = _array(model, ("C13",), shape, "C13")
    c33 = _array(model, ("C33",), shape, "C33")
    c44 = _array(model, ("C44",), shape, "C44")
    c66 = _array(model, ("C66",), shape, "C66")
    validation = validate_vti_material(rho, c11, c13, c33, c44, c66)
    validation["cfl_dt_limit_s"] = conservative_cfl_limit(
        dx, dz, validation["conservative_vmax_m_s"], order
    )

    source_data = _load(directory / "source.mat")
    dt = _scalar(source_data, "dt")
    total_time = _scalar(source_data, "T")
    nt = _scalar(source_data, "N", int)
    source = np.asarray(source_data["src"], dtype=np.float64).reshape(-1)
    if source.size < nt:
        source = np.pad(source, (0, nt - source.size))
    source = np.ascontiguousarray(source[:nt])
    if dt <= 0.0 or nt < 4 or not np.all(np.isfinite(source)):
        raise ValueError(f"invalid source sampling: dt={dt}, N={nt}")
    if dt > validation["cfl_dt_limit_s"]:
        raise ValueError(
            f"unstable time step: dt={dt:.8e} exceeds conservative limit "
            f"{validation['cfl_dt_limit_s']:.8e} for order={order}"
        )
    default_source_kind = _text(source_data.get("source_kind"), "explosive").lower()
    if default_source_kind not in SOURCE_KINDS:
        raise ValueError(f"unsupported source_kind in source.mat: {default_source_kind}")
    source_name = _text(source_data.get("src_name"), "unknown")
    source_f0_hz = _optional_scalar(source_data, "f0")
    source_t0_s = _optional_scalar(source_data, "t0")
    source_scale = _optional_scalar(source_data, "src_scale")
    if source_f0_hz is not None and (not np.isfinite(source_f0_hz) or source_f0_hz <= 0.0):
        raise ValueError(f"invalid source f0={source_f0_hz}")
    if source_t0_s is not None and (not np.isfinite(source_t0_s) or source_t0_s < 0.0):
        raise ValueError(f"invalid source t0={source_t0_s}")
    if source_scale is not None and not np.isfinite(source_scale):
        raise ValueError(f"invalid source scale={source_scale}")

    boundary = _load(directory / "BC.mat")
    nab = _scalar(boundary, "nAB", int)
    bc_type = _text(boundary.get("BCtype"), "topABC")
    pml_name = _text(boundary.get("BCname"), "PML")
    pml_reflection_coefficient = _optional_scalar(boundary, "reflection_coefficient")
    pml_polynomial_power = _optional_scalar(boundary, "polynomial_power")
    if bc_type != "topABC":
        raise NotImplementedError("the accelerated 2-D solver currently supports split-PML topABC only")
    if nab < 1 or 2 * nab >= min(nx, nz):
        raise ValueError(f"invalid PML nAB={nab} for nx/nz={nx}/{nz}")
    if pml_reflection_coefficient is not None and not 0.0 < pml_reflection_coefficient < 1.0:
        raise ValueError(f"invalid PML reflection_coefficient={pml_reflection_coefficient}")
    if pml_polynomial_power is not None and (
        not np.isfinite(pml_polynomial_power) or pml_polynomial_power <= 0.0
    ):
        raise ValueError(f"invalid PML polynomial_power={pml_polynomial_power}")
    bc = boundary["BC"]
    pmlx = np.ascontiguousarray(np.asarray(_struct_field(bc, "pmlx"), dtype=np.float64))
    pmlz = np.ascontiguousarray(np.asarray(_struct_field(bc, "pmlz"), dtype=np.float64))
    for name, array in (("pmlx", pmlx), ("pmlz", pmlz)):
        if array.shape != shape or not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"invalid {name}: shape={array.shape}, expected={shape}")

    # Geometry files store MATLAB indices; all solver coordinates are zero-based.
    source_geometry = _load(directory / "geometry_src.mat")
    src_x = int(np.asarray(source_geometry["snx_vec"]).reshape(-1)[0]) - 1
    src_z = int(np.asarray(source_geometry["snz_vec"]).reshape(-1)[0]) - 1
    receiver_geometry = _load(directory / "geometry_rec.mat")
    rec_x = np.asarray(receiver_geometry["rnx_vec"], dtype=np.int64).reshape(-1) - 1
    rec_z = np.asarray(receiver_geometry["rnz_vec"], dtype=np.int64).reshape(-1) - 1
    if not (0 <= src_x < nx and 0 <= src_z < nz):
        raise ValueError(f"source [x,z]=[{src_x},{src_z}] is outside nx/nz=[{nx},{nz}]")
    half = order // 2
    if not (half <= src_x < nx - half and half <= src_z < nz - half):
        raise ValueError(
            f"source [x,z]=[{src_x},{src_z}] lies outside the active order-{order} FD region"
        )
    valid_receivers = (rec_x >= 0) & (rec_x < nx) & (rec_z >= 0) & (rec_z < nz)
    if rec_x.size == 0 or rec_x.size != rec_z.size or not np.all(valid_receivers):
        raise ValueError("receiver coordinates are empty, mismatched, or outside the model")

    bx, bz, c55s = derive_staggered(rho, c44)
    return VTI2DCase(
        case_dir=directory,
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        rho=rho,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        bx=bx,
        bz=bz,
        c55s=c55s,
        pmlx=pmlx,
        pmlz=pmlz,
        dt=dt,
        total_time=total_time,
        nt=nt,
        source=source,
        source_name=source_name,
        source_f0_hz=source_f0_hz,
        source_t0_s=source_t0_s,
        source_scale=source_scale,
        default_source_kind=default_source_kind,
        src_x=src_x,
        src_z=src_z,
        rec_x=rec_x,
        rec_z=rec_z,
        nab=nab,
        bc_type=bc_type,
        pml_name=pml_name,
        pml_reflection_coefficient=pml_reflection_coefficient,
        pml_polynomial_power=pml_polynomial_power,
        validation=validation,
    )

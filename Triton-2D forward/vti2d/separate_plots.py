"""Save separate Vx/Vz snapshots, near-source traces, and receiver gathers.

Input diagnostics are checked for finite values. Spatial axes use metres
and receiver layouts use physical coordinates or cumulative distance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def _finite_array(value, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        bad = tuple(int(index) for index in np.argwhere(~np.isfinite(array))[0])
        raise ValueError(f"{name} contains NaN/Inf at index={bad}")
    return array


def _finite_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.ndim > 2 or (array.ndim == 2 and 1 not in array.shape):
        raise ValueError(f"{name} must be a vector, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        bad = tuple(int(index) for index in np.argwhere(~np.isfinite(array))[0])
        raise ValueError(f"{name} contains NaN/Inf at index={bad}")
    return array.reshape(-1)


def _finite_scalar(value, name: str) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar, got shape={np.asarray(value).shape}")
    if not np.isfinite(array[0]):
        raise ValueError(f"{name} must be finite")
    return float(array[0])


def _physical_extent(case) -> tuple[float, float, float, float]:
    if case.dx <= 0.0 or case.dz <= 0.0:
        raise ValueError(f"dx and dz must be positive, got dx={case.dx}, dz={case.dz}")
    return 0.0, (case.nx - 1) * case.dx, (case.nz - 1) * case.dz, 0.0


def _positive_limit(*arrays: np.ndarray) -> float:
    return max(*(float(np.max(np.abs(array))) for array in arrays), 1.0e-30)


def _snapshot_size(case) -> tuple[float, float]:
    width_m = max((case.nx - 1) * case.dx, case.dx)
    height_m = max((case.nz - 1) * case.dz, case.dz)
    ratio = width_m / height_m
    width = min(14.0, max(7.2, 7.2 * ratio))
    height = min(11.0, max(5.4, width / ratio))
    return width, height


def _save_snapshot(
    field: np.ndarray,
    component: str,
    case,
    diagnostics: dict,
    output: Path,
    stem: str,
    cmap: str,
    dpi: int,
    norm: Normalize,
) -> Path:
    figure, axis = plt.subplots(figsize=_snapshot_size(case), constrained_layout=True)
    image = axis.imshow(
        field,
        origin="upper",
        extent=_physical_extent(case),
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    source_x_m = case.src_x * case.dx
    source_z_m = case.src_z * case.dz
    axis.scatter(
        source_x_m, source_z_m, marker="*", s=105,
        c="#f6c445", edgecolors="#202124", linewidths=0.8, zorder=3,
    )
    step = int(_finite_scalar(diagnostics["snapshot_step"], "snapshot_step"))
    time_s = _finite_scalar(diagnostics["snapshot_time"], "snapshot_time")
    axis.set_title(f"{component} wavefield snapshot\nstep={step}, time={time_s:.6g} s")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Z (m, depth)")
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.025)
    colorbar.set_label(f"{component} particle velocity (m/s)")
    path = output / f"{stem}_snapshot_{component.lower()}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _save_near_trace(
    trace: np.ndarray,
    component: str,
    case,
    diagnostics: dict,
    output: Path,
    stem: str,
    dpi: int,
) -> Path:
    record_stride = int(_finite_scalar(diagnostics.get("record_stride", 1), "record_stride"))
    if record_stride < 1:
        raise ValueError("record_stride must be positive")
    time_s = np.arange(trace.size, dtype=np.float64) * case.dt * record_stride
    near_x = int(_finite_scalar(diagnostics["near_rec_x"], "near_rec_x"))
    near_z = int(_finite_scalar(diagnostics["near_rec_z"], "near_rec_z"))
    near_x_m, near_z_m = near_x * case.dx, near_z * case.dz
    color = "#2457a6" if component == "Vx" else "#d97706"
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    axis.plot(time_s, trace, color=color, linewidth=1.15)
    axis.axhline(0.0, color="#555555", linewidth=0.75)
    axis.set_title(
        f"Near-source receiver {component}\n"
        f"index [x,z]=[{near_x},{near_z}], position [X,Z]=[{near_x_m:.6g},{near_z_m:.6g}] m"
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(f"{component} particle velocity (m/s)")
    axis.grid(True, color="#d9d9d9", linewidth=0.55, alpha=0.75)
    path = output / f"{stem}_near_source_{component.lower()}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _receiver_coordinate(case) -> tuple[np.ndarray, str, str]:
    x_m = np.asarray(case.rec_x, dtype=np.float64) * case.dx
    z_m = np.asarray(case.rec_z, dtype=np.float64) * case.dz
    horizontal = np.allclose(z_m, z_m[0])
    monotonic_x = np.all(np.diff(x_m) > 0.0) or np.all(np.diff(x_m) < 0.0)
    if horizontal and monotonic_x:
        return x_m, "Receiver X (m)", f"receiver depth Z={z_m[0]:.6g} m"
    segment = np.hypot(np.diff(x_m), np.diff(z_m))
    distance = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(segment)))
    return distance, "Receiver-line distance (m)", "distance uses both dx and dz"


def _save_gather(
    gather: np.ndarray,
    component: str,
    case,
    diagnostics: dict,
    output: Path,
    stem: str,
    cmap: str,
    dpi: int,
    norm: Normalize,
) -> Path:
    record_stride = int(_finite_scalar(diagnostics.get("record_stride", 1), "record_stride"))
    time_s = np.arange(gather.shape[0], dtype=np.float64) * case.dt * record_stride
    receiver_coordinate, coordinate_label, coordinate_note = _receiver_coordinate(case)
    if gather.shape[1] != receiver_coordinate.size:
        raise ValueError(
            f"{component} gather receiver count={gather.shape[1]}, "
            f"geometry receiver count={receiver_coordinate.size}"
        )
    figure, axis = plt.subplots(figsize=(10.8, 6.6), constrained_layout=True)
    image = axis.pcolormesh(
        receiver_coordinate, time_s, gather,
        cmap=cmap, norm=norm, shading="auto", rasterized=True,
    )
    axis.invert_yaxis()
    axis.set_title(f"Receiver {component} seismic record\n{coordinate_note}")
    axis.set_xlabel(coordinate_label)
    axis.set_ylabel("Time (s)")
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.88, pad=0.025)
    colorbar.set_label(f"{component} particle velocity (m/s)")
    path = output / f"{stem}_receiver_gather_{component.lower()}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def save_separate_component_figures(
    result,
    case,
    output_dir,
    stem,
    cmap="seismic",
    dpi=180,
) -> dict[str, Path]:
    """Save six separate Vx/Vz snapshot, near-receiver and gather figures.

    Snapshot coordinates use X=x*dx and Z=z*dz.  A horizontal receiver line
    uses physical X coordinates; other receiver layouts use cumulative line
    distance computed from both dx and dz.
    """
    if int(dpi) < 1:
        raise ValueError("dpi must be positive")
    diagnostics = result["diagnostics"]
    vx = _finite_array(diagnostics["snapshot_vx"], "snapshot_vx", 2)
    vz = _finite_array(diagnostics["snapshot_vz"], "snapshot_vz", 2)
    sx = _finite_array(result["Sx"], "Sx/Vx gather", 2)
    sz = _finite_array(result["Sz"], "Sz/Vz gather", 2)
    near_vx = _finite_vector(diagnostics["near_Sx"], "near_Sx/Vx")
    near_vz = _finite_vector(diagnostics["near_Sz"], "near_Sz/Vz")
    if vx.shape != case.shape or vz.shape != case.shape:
        raise ValueError(f"snapshot shape must be {case.shape}, got Vx={vx.shape}, Vz={vz.shape}")
    if sx.shape != sz.shape:
        raise ValueError(f"Vx/Vz gather shape mismatch: {sx.shape} != {sz.shape}")
    if near_vx.size != sx.shape[0] or near_vz.size != sx.shape[0]:
        raise ValueError("near-source trace length must equal receiver record length")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_norm = Normalize(-_positive_limit(vx, vz), _positive_limit(vx, vz))
    gather_norm = Normalize(-_positive_limit(sx, sz), _positive_limit(sx, sz))
    paths = {
        "snapshot_vx": _save_snapshot(vx, "Vx", case, diagnostics, output, stem, cmap, dpi, snapshot_norm),
        "snapshot_vz": _save_snapshot(vz, "Vz", case, diagnostics, output, stem, cmap, dpi, snapshot_norm),
        "near_source_vx": _save_near_trace(near_vx, "Vx", case, diagnostics, output, stem, dpi),
        "near_source_vz": _save_near_trace(near_vz, "Vz", case, diagnostics, output, stem, dpi),
        "receiver_gather_vx": _save_gather(sx, "Vx", case, diagnostics, output, stem, cmap, dpi, gather_norm),
        "receiver_gather_vz": _save_gather(sz, "Vz", case, diagnostics, output, stem, cmap, dpi, gather_norm),
    }
    return paths

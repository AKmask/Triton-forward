"""Render overview figures for 2-D wavefields and receiver records.

The noninteractive Matplotlib backend writes PNG files without opening
windows; spatial coordinates are converted from grid indices to metres.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def _extent(case):
    return (0.0, (case.nx - 1) * case.dx, (case.nz - 1) * case.dz, 0.0)


def save_wavefield_figure(result, case, output_dir, stem, cmap="seismic", dpi=180) -> Path:
    """Save Vx, Vz, and velocity magnitude at the selected snapshot step."""
    diagnostics = result["diagnostics"]
    vx = np.asarray(diagnostics["snapshot_vx"], dtype=np.float64)
    vz = np.asarray(diagnostics["snapshot_vz"], dtype=np.float64)
    magnitude = np.sqrt(vx * vx + vz * vz)
    signed_limit = max(float(np.max(np.abs(vx))), float(np.max(np.abs(vz))), 1.0e-30)
    magnitude_limit = max(float(np.max(magnitude)), 1.0e-30)
    signed_norm = Normalize(-signed_limit, signed_limit)
    magnitude_norm = Normalize(0.0, magnitude_limit)
    source_x = case.src_x * case.dx
    source_z = case.src_z * case.dz
    fields = ((vx, "vx", signed_norm), (vz, "vz", signed_norm), (magnitude, "Velocity magnitude", magnitude_norm))

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), constrained_layout=True, sharex=True, sharey=True)
    images = []
    for axis, (field, title, norm) in zip(axes, fields):
        image = axis.imshow(
            field,
            origin="upper",
            extent=_extent(case),
            aspect="equal",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        images.append(image)
        axis.scatter(source_x, source_z, marker="*", s=95, c="#ffd166", edgecolors="#202124", linewidths=0.8)
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        axis.grid(False)
    axes[0].set_ylabel("Z (m, depth)")
    step = int(diagnostics["snapshot_step"])
    time_s = float(diagnostics["snapshot_time"])
    figure.suptitle(f"2-D Hess VTI wavefield, step={step}, time={time_s:.6g} s", fontsize=14)
    signed_colorbar = figure.colorbar(images[0], ax=axes[:2], shrink=0.82, pad=0.02)
    signed_colorbar.set_label("Particle velocity (m/s)")
    magnitude_colorbar = figure.colorbar(images[2], ax=axes[2], shrink=0.82, pad=0.02)
    magnitude_colorbar.set_label("Velocity magnitude (m/s)")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}_wavefield.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def save_receiver_figure(result, case, output_dir, stem, cmap="seismic", dpi=180) -> Path:
    """Save Vx/Vz receiver gathers and near-source traces in one figure."""
    sx = np.asarray(result["Sx"], dtype=np.float64)
    sz = np.asarray(result["Sz"], dtype=np.float64)
    near_sx = np.asarray(result["diagnostics"]["near_Sx"], dtype=np.float64).reshape(-1)
    near_sz = np.asarray(result["diagnostics"]["near_Sz"], dtype=np.float64).reshape(-1)
    limit = max(float(np.max(np.abs(sx))), float(np.max(np.abs(sz))), 1.0e-30)
    norm = Normalize(-limit, limit)
    record_stride = int(result["diagnostics"].get("record_stride", 1))
    time_axis = np.arange(sx.shape[0], dtype=np.float64) * case.dt * record_stride
    receiver_axis = np.arange(case.nrec)
    extent = (
        float(receiver_axis[0]) if receiver_axis.size else 0.0,
        float(receiver_axis[-1]) if receiver_axis.size > 1 else 1.0,
        float(time_axis[-1]),
        float(time_axis[0]),
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)
    images = []
    for axis, gather, title in ((axes[0, 0], sx, "Receiver vx"), (axes[0, 1], sz, "Receiver vz")):
        image = axis.imshow(gather, origin="upper", aspect="auto", extent=extent, cmap=cmap, norm=norm)
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("Receiver number")
        axis.set_ylabel("Time (s)")
    colorbar = figure.colorbar(images[0], ax=axes[0, :], shrink=0.84, pad=0.02)
    colorbar.set_label("Particle velocity (m/s)")
    axes[1, 0].plot(time_axis, near_sx, color="#2457a6", linewidth=1.1, label="near-source vx")
    axes[1, 1].plot(time_axis, near_sz, color="#d97706", linewidth=1.1, label="near-source vz")
    for axis in axes[1, :]:
        axis.axhline(0.0, color="#555555", linewidth=0.7)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Particle velocity (m/s)")
        axis.legend(loc="upper right")
        axis.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
    figure.suptitle("2-D Hess VTI receiver responses", fontsize=14)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}_receivers.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path

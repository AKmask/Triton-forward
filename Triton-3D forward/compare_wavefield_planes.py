from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
from scipy.io import loadmat


PLANES = ("xy", "xz", "yz")
COMPONENTS = ("vx", "vy", "vz")
RESULT_FIELDS = {
    *(f"snapshot_{c}_{p}" for c in COMPONENTS for p in PLANES),
    "snapshot_step", "snapshot_time", "src_x", "src_y", "src_z",
    "nx", "ny", "nz", "dx", "dy", "dz", "dt", "N", "M", "dN_SS",
    "near_rec_x", "near_rec_y", "near_rec_z", "near_Sx", "near_Sy", "near_Sz",
    "material_model", "source_kind",
}


def load_result(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            # Do not load unrelated object metadata (or enable unsafe pickle).
            return {name: np.asarray(data[name]) for name in data.files if name in RESULT_FIELDS}
    if path.suffix.lower() == ".mat":
        return {
            name: np.asarray(value)
            for name, value in loadmat(path, variable_names=list(RESULT_FIELDS), squeeze_me=False).items()
            if not name.startswith("__")
        }
    raise ValueError(f"Only .mat and .npz are supported: {path}")


def scalar(data: dict, name: str, cast=float):
    if name not in data:
        raise KeyError(f"Missing required variable: {name}")
    values = np.asarray(data[name])
    if values.size != 1:
        raise ValueError(f"{name} must be a scalar")
    value = float(values.reshape(-1)[0])
    if not np.isfinite(value) or (cast is int and value != int(value)):
        raise ValueError(f"Invalid {name}: {value}")
    return cast(value)


def vector_or_component(data: dict, plane: str, component: str) -> np.ndarray:
    if component == "magnitude":
        arrays = [np.asarray(data[f"snapshot_{c}_{plane}"], dtype=np.float64) for c in COMPONENTS]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError(f"Inconsistent component shapes on {plane.upper()} plane")
        return np.hypot(np.hypot(arrays[0], arrays[1]), arrays[2])
    return np.asarray(data[f"snapshot_{component}_{plane}"], dtype=np.float64)


def relative_l2(reference: np.ndarray, difference: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm(difference.ravel()))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else None
    return numerator / denominator


def error_metrics(reference: np.ndarray, test: np.ndarray) -> dict[str, float | None]:
    difference = test - reference
    reference_flat = reference.ravel()
    test_flat = test.ravel()
    if reference_flat.size > 1 and np.std(reference_flat) > 0 and np.std(test_flat) > 0:
        correlation = float(np.corrcoef(reference_flat, test_flat)[0, 1])
    else:
        correlation = None
    return {
        "reference_max_abs": float(np.max(np.abs(reference))),
        "test_max_abs": float(np.max(np.abs(test))),
        "difference_max_abs": float(np.max(np.abs(difference))),
        "difference_relative_l2": relative_l2(reference, difference),
        "correlation": correlation,
    }


def validate_alignment(reference: dict, test: dict, *, receivers: bool = False) -> list[str]:
    integer_names = (
        "snapshot_step", "src_x", "src_y", "src_z",
        "nx", "ny", "nz",
    )
    mismatches = []
    for name in integer_names:
        a, b = scalar(reference, name, int), scalar(test, name, int)
        if a != b:
            mismatches.append(f"{name}: {a} != {b}")
    for name in ("dt", "dx", "dy", "dz", "snapshot_time"):
        a, b = scalar(reference, name), scalar(test, name)
        if not np.isclose(a, b, rtol=1e-12, atol=1e-15):
            mismatches.append(f"{name}: {a} != {b}")
    warnings = [
        "Snapshot files alone cannot prove identical stiffness, density, boundary or source wavelet. "
        "Run both solvers on the same case and source kind."
    ]
    for name in ("M", "N"):
        if name in reference and name in test:
            if scalar(reference, name, int) != scalar(test, name, int):
                mismatches.append(f"{name} differs")
        else:
            warnings.append(f"{name} is missing; its alignment was not verified.")
    for name in ("material_model", "source_kind"):
        if name in reference and name in test:
            a = "".join(np.asarray(reference[name]).astype(str).ravel()).strip()
            b = "".join(np.asarray(test[name]).astype(str).ravel()).strip()
            if a.lower() != b.lower():
                mismatches.append(f"{name}: {a!r} != {b!r}")
        else:
            warnings.append(f"{name} is missing; it was not verified from the results.")
    for label, data in (("Direct", reference), ("10 VTI", test)):
        nx, ny, nz = (scalar(data, name, int) for name in ("nx", "ny", "nz"))
        if min(nx, ny, nz) < 2:
            raise ValueError(f"{label}: grid dimensions must be at least 2")
        if any(scalar(data, name) <= 0 for name in ("dx", "dy", "dz", "dt")):
            raise ValueError(f"{label}: grid spacing and dt must be positive")
        if not all(0 <= scalar(data, f"src_{axis}", int) < size
                   for axis, size in zip("xyz", (nx, ny, nz))):
            raise ValueError(f"{label}: source is outside the grid")
        if scalar(data, "snapshot_step", int) < 1 or not np.isclose(
            scalar(data, "snapshot_time"),
            (scalar(data, "snapshot_step", int) - 1) * scalar(data, "dt"),
            rtol=1e-12, atol=1e-15,
        ):
            raise ValueError(f"{label}: inconsistent snapshot_step/time/dt")
        for plane, shape in zip(PLANES, ((ny, nx), (nz, nx), (nz, ny))):
            for component in COMPONENTS:
                key = f"snapshot_{component}_{plane}"
                if key not in data:
                    raise KeyError(f"Missing {key}; use the consolidated solver output, not an old record-only file")
                array = np.asarray(data[key])
                if array.shape != shape:
                    raise ValueError(f"{label} {key}: expected {shape}, got {array.shape}; no automatic transpose")
                if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array) or not np.isfinite(array).all():
                    raise ValueError(f"{label} {key}: data must be real and finite")
    if receivers:
        for name in ("near_rec_x", "near_rec_y", "near_rec_z", "dN_SS"):
            if scalar(reference, name, int) != scalar(test, name, int):
                mismatches.append(f"{name} differs")
        if scalar(reference, "dN_SS", int) < 1:
            raise ValueError("dN_SS must be positive")
        for component in ("Sx", "Sy", "Sz"):
            a = np.asarray(reference[f"near_{component}"]).reshape(-1)
            b = np.asarray(test[f"near_{component}"]).reshape(-1)
            if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Invalid or mismatched near_{component}")
    if mismatches:
        raise ValueError("The two outputs are not spatially/temporally aligned:\n  " + "\n  ".join(mismatches))
    return warnings


def plane_extent_and_labels(data: dict, plane: str, shape: tuple[int, int]):
    dx, dy, dz = (scalar(data, name) for name in ("dx", "dy", "dz"))
    if plane == "xy":
        return (-dx/2, (shape[1] - 0.5)*dx, (shape[0] - 0.5)*dy, -dy/2), "X (m)", "Y (m)"
    if plane == "xz":
        return (-dx/2, (shape[1] - 0.5)*dx, (shape[0] - 0.5)*dz, -dz/2), "X (m)", "Z (m)"
    return (-dy/2, (shape[1] - 0.5)*dy, (shape[0] - 0.5)*dz, -dz/2), "Y (m)", "Z (m)"


def source_marker(data: dict, plane: str) -> tuple[float, float]:
    x = scalar(data, "src_x", int) * scalar(data, "dx")
    y = scalar(data, "src_y", int) * scalar(data, "dy")
    z = scalar(data, "src_z", int) * scalar(data, "dz")
    return {"xy": (x, y), "xz": (x, z), "yz": (y, z)}[plane]


def create_plane_figure(reference: dict, test: dict, component: str,
                        labels=("PyTorch Direct", "10 VTI"), *, cmap="seismic", show_source=False):
    """Nine panels share one symmetric normalization; no smoothing or resampling."""
    if component not in (*COMPONENTS, "magnitude"):
        raise ValueError(f"Unsupported component: {component}")
    arrays = {}
    metrics = {}
    for plane in PLANES:
        ref = vector_or_component(reference, plane, component)
        candidate = vector_or_component(test, plane, component)
        if ref.shape != candidate.shape:
            raise ValueError(f"{plane.upper()} shape mismatch: {ref.shape} != {candidate.shape}")
        difference = candidate - ref
        if not all(np.isfinite(a).all() for a in (ref, candidate, difference)):
            raise ValueError(f"Non-finite values on {plane.upper()}")
        arrays[plane] = (candidate, ref, difference)
        metrics[plane] = error_metrics(ref, candidate)
    # Include residuals so a large difference is never silently clipped.
    data_limit = max(float(np.max(np.abs(a))) for panels in arrays.values() for a in panels)
    common_limit = data_limit if data_limit > 0 else 1.0
    norm = Normalize(vmin=-common_limit, vmax=common_limit)
    figure, axes = plt.subplots(3, 3, figsize=(12.8, 11.8), sharey="row", constrained_layout=True)
    for row, (plane, fixed_axis) in enumerate(zip(PLANES, "zyx")):
        extent, xlabel, ylabel = plane_extent_and_labels(reference, plane, arrays[plane][0].shape)
        coordinate = scalar(reference, f"src_{fixed_axis}", int) * scalar(reference, f"d{fixed_axis}")
        for column, array in enumerate(arrays[plane]):
            axis = axes[row, column]
            image = axis.imshow(array, origin="upper", extent=extent, aspect="equal",
                                interpolation="nearest", cmap=cmap, norm=norm)
            axis.text(0.025, 0.96, f"({chr(97 + row*3 + column)})", transform=axis.transAxes,
                      va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
            if show_source:
                axis.scatter(*source_marker(reference, plane), marker="*", s=48, facecolors="none",
                             edgecolors="#111827", linewidths=0.8)
            if row == 0:
                axis.set_title((labels[1], labels[0], f"Difference ({labels[1]} − {labels[0]})")[column],
                               fontsize=11, pad=10)
            axis.set_xlabel(xlabel, fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{plane.upper()} | {fixed_axis.upper()} = {coordinate:g} m\n{ylabel}", fontsize=10)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            axis.tick_params(labelsize=9)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#6b7280")
        relative = metrics[plane]["difference_relative_l2"]
        rel_text = f"{relative:.2e}" if relative is not None else "undefined (Direct=0)"
        axes[row, 2].set_xlabel(
            f"{xlabel}\nrel. L2 = {rel_text}  |  max |Δ| = {metrics[plane]['difference_max_abs']:.2e}", fontsize=9
        )
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.025, aspect=35)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    colorbar.formatter = formatter
    colorbar.update_ticks()
    colorbar.set_label("Velocity amplitude (simulation units)", fontsize=10)
    step = scalar(reference, "snapshot_step", int)
    time_s = scalar(reference, "snapshot_time")
    quantity = "|v| (magnitude difference, not vector-error norm)" if component == "magnitude" else component.upper()
    figure.suptitle(
        f"{quantity} | source-plane snapshots | t = {time_s:.6g} s | step = {step}\n"
        "Same symmetric color scale for all nine panels; differences are not amplified", fontsize=13
    )
    return figure, {"planes": metrics, "color_limit": common_limit, "all_zero": data_limit == 0}


def plot_planes(reference: dict, test: dict, component: str, labels: tuple[str, str], output: Path,
                *, cmap="seismic", dpi=300, show_source=False):
    figure, details = create_plane_figure(reference, test, component, labels, cmap=cmap, show_source=show_source)
    try:
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    return details["planes"]


def plot_receivers(reference: dict, test: dict, labels: tuple[str, str], output: Path):
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    dt = scalar(reference, "dt")
    stride = scalar(reference, "dN_SS", int)
    metrics = {}
    colors = ("#1f5a99", "#d97706")
    for row, component in enumerate(("Sx", "Sy", "Sz")):
        ref = np.asarray(reference[f"near_{component}"], dtype=np.float64).reshape(-1)
        candidate = np.asarray(test[f"near_{component}"], dtype=np.float64).reshape(-1)
        if ref.shape != candidate.shape:
            raise ValueError(f"near_{component} shape mismatch: {ref.shape} != {candidate.shape}")
        difference = candidate - ref
        metrics[component.lower()] = error_metrics(ref, candidate)
        time_axis = np.arange(ref.size, dtype=np.float64) * dt * stride
        response_axis, difference_axis = axes[row]
        response_axis.plot(time_axis, ref, color=colors[0], linewidth=1.2, label=labels[0])
        response_axis.plot(time_axis, candidate, color=colors[1], linewidth=1.0, linestyle="--", label=labels[1])
        response_axis.set_title(f"Near-source receiver {component}")
        response_axis.set_ylabel("velocity")
        response_axis.grid(True, color="#d9d9d9", linewidth=0.5)
        response_axis.legend(loc="upper right", frameon=False)
        response_ylim = response_axis.get_ylim()
        difference_axis.plot(time_axis, difference, color="#4b5563", linewidth=1.0)
        difference_axis.axhline(0.0, color="black", linewidth=0.6)
        # Use exactly the same vertical range as the original response panel.
        difference_axis.set_ylim(response_ylim)
        difference_axis.set_title(f"{component} difference ({labels[1]} - {labels[0]})")
        difference_axis.set_ylabel("velocity difference")
        difference_axis.grid(True, color="#d9d9d9", linewidth=0.5)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    receiver_xyz = tuple(scalar(reference, f"near_rec_{axis}", int) for axis in "xyz")
    figure.suptitle(f"Near-source receiver response | zero-based xyz={receiver_xyz}", fontsize=15)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare three source-plane snapshots and a near-source receiver")
    parser.add_argument("--fd", required=True, help="FDwave3D-PyTorch-Direct .mat/.npz output")
    parser.add_argument("--triton", required=True, help="10_region_fused_splitpml_vti .mat/.npz output")
    parser.add_argument("--component", choices=["all", "magnitude", *COMPONENTS], default="vz")
    parser.add_argument("--cmaps", nargs="+", choices=["seismic", "gray", "RdBu_r"], default=["seismic", "gray"])
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-source", action="store_true", help="Mark the source in each panel")
    parser.add_argument("--skip-receivers", action="store_true", help="Compare snapshots only")
    parser.add_argument("--overwrite", action="store_true", help="Replace named comparison outputs")
    parser.add_argument("--fd-label", default="PyTorch Direct")
    parser.add_argument("--triton-label", default="10 VTI")
    parser.add_argument("--output-dir", default="comparison_wavefield_planes")
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    reference = load_result(args.fd)
    test = load_result(args.triton)
    receiver_keys = {"near_Sx", "near_Sy", "near_Sz", "near_rec_x", "near_rec_y", "near_rec_z", "dN_SS"}
    include_receivers = not args.skip_receivers and receiver_keys <= reference.keys() and receiver_keys <= test.keys()
    warnings = validate_alignment(reference, test, receivers=include_receivers)
    output_dir = Path(args.output_dir).expanduser().resolve()
    labels = (args.fd_label, args.triton_label)
    components = COMPONENTS if args.component == "all" else (args.component,)
    cmaps, formats = tuple(dict.fromkeys(args.cmaps)), tuple(dict.fromkeys(args.formats))
    plane_paths = [output_dir / f"wavefield_{component}_{cmap}.{fmt}"
                   for component in components for cmap in cmaps for fmt in formats]
    receiver_path = output_dir / "near_source_receiver_comparison.png"
    metrics_path = output_dir / "comparison_metrics.json"
    paths = [*plane_paths, metrics_path] + ([receiver_path] if include_receivers else [])
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist: {existing}. Use --overwrite only if replacement is intended.")
    if {Path(args.fd).resolve(), Path(args.triton).resolve()} & set(paths):
        raise ValueError("An output path would overwrite an input file")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "inputs": {"fd": str(Path(args.fd).resolve()), "triton": str(Path(args.triton).resolve())},
        "components": list(components),
        "difference_definition": "10 VTI - PyTorch Direct; signed, without amplification",
        "color_scale": "one symmetric scale per component, shared by all planes and originals/differences",
        "warnings": warnings,
        "snapshot_step": scalar(reference, "snapshot_step", int),
        "snapshot_time": scalar(reference, "snapshot_time"),
        "snapshots": {},
        "figures": [str(path) for path in plane_paths],
    }
    for component in components:
        for cmap in cmaps:
            figure, details = create_plane_figure(reference, test, component, labels,
                                                  cmap=cmap, show_source=args.show_source)
            try:
                for fmt in formats:
                    figure.savefig(output_dir / f"wavefield_{component}_{cmap}.{fmt}",
                                   dpi=args.dpi, bbox_inches="tight", facecolor="white")
            finally:
                plt.close(figure)
            metrics["snapshots"][component] = details
    if include_receivers:
        metrics["near_receiver"] = plot_receivers(reference, test, labels, receiver_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "wavefield_figures": [str(path) for path in plane_paths],
        "receiver_figure": str(receiver_path) if include_receivers else None,
        "metrics": str(metrics_path),
        "summary": metrics,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

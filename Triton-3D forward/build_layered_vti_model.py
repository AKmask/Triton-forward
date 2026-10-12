"""Build a horizontally layered 3-D VTI case from a JSON configuration.

MAT arrays use [z, x, y] order. Layer intervals are half-open grid-index
ranges; exported MATLAB geometry uses one-based coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vti10.runtime_paths import configure_runtime_cache

# Configure caches before third-party libraries initialize their runtimes.
CACHE_ROOT = configure_runtime_cache()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from scipy.io import savemat

from vti10.vti_stiffness import validate_vti_material


MATERIAL_FIELDS = ("rho", "C11", "C13", "C33", "C44", "C66")
CASE_FILENAMES = (
    "model.mat",
    "source.mat",
    "BC.mat",
    "geometry_src.mat",
    "geometry_rec.mat",
    "model_config_used.json",
    "model_summary.json",
    "run_forward.ps1",
)
PLOT_MODES = ("combined", "model", "parameters")


def _mapping(parent: dict, name: str) -> dict:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field {name} must be a JSON object")
    return value


def _finite_float(parent: dict, name: str) -> float:
    if name not in parent:
        raise ValueError(f"Configuration is missing parameter {name}")
    value = float(parent[name])
    if not math.isfinite(value):
        raise ValueError(f"Parameter {name} must be finite, got {value}")
    return value


def _positive_float(parent: dict, name: str) -> float:
    value = _finite_float(parent, name)
    if value <= 0.0:
        raise ValueError(f"Parameter {name} must be positive, got {value}")
    return value


def _integer(parent: dict, name: str, minimum: int | None = None) -> int:
    if name not in parent:
        raise ValueError(f"Configuration is missing parameter {name}")
    raw = parent[name]
    value = int(raw)
    if isinstance(raw, float) and raw != value:
        raise ValueError(f"Parameter {name} must be an integer, got {raw}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Parameter {name} must be at least {minimum}, got {value}")
    return value


def load_config(path: str | Path) -> tuple[dict, Path]:
    """Read the JSON configuration and retain its resolved path."""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object")
    return config, path


def resolve_output_dir(config: dict, config_path: Path, override: str | None) -> Path:
    """Resolve the configured or overridden case path relative to the JSON file."""
    raw = override if override is not None else config.get("output_case_dir")
    if not raw:
        raise ValueError("Set output_case_dir in the configuration or specify --output-dir")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def parse_receivers(receiver_config: dict, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Build zero-based [x, y, z] points from an axis line or an explicit list."""
    mode = str(receiver_config.get("mode", "axis_line")).lower()
    nz, ny, nx = shape_zyx
    if mode == "explicit":
        points = np.asarray(receiver_config.get("points_xyz", []), dtype=np.int64)
        if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] == 0:
            raise ValueError("Explicit receivers require non-empty points_xyz in the form [[x,y,z],...]")
    elif mode == "axis_line":
        axis = str(receiver_config.get("axis", "x")).lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("Receiver axis must be x, y, or z")
        start = _integer(receiver_config, "start", 0)
        stop = _integer(receiver_config, "stop", 0)
        step = _integer(receiver_config, "step", 1)
        if start > stop:
            raise ValueError("Receiver line start must not exceed stop")
        coordinates = np.arange(start, stop + 1, step, dtype=np.int64)
        fixed = {
            "x": _integer(receiver_config, "fixed_x", 0),
            "y": _integer(receiver_config, "fixed_y", 0),
            "z": _integer(receiver_config, "fixed_z", 0),
        }
        points = np.empty((coordinates.size, 3), dtype=np.int64)
        for column, coordinate_name in enumerate(("x", "y", "z")):
            points[:, column] = coordinates if coordinate_name == axis else fixed[coordinate_name]
    else:
        raise ValueError("receivers.mode must be axis_line or explicit")

    valid = (
        (points[:, 0] >= 0) & (points[:, 0] < nx)
        & (points[:, 1] >= 0) & (points[:, 1] < ny)
        & (points[:, 2] >= 0) & (points[:, 2] < nz)
    )
    if not np.all(valid):
        index = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            f"Receiver {index} at {points[index].tolist()} is out of bounds; "
            f"valid x/y/z ranges are [0,{nx-1}]/[0,{ny-1}]/[0,{nz-1}]"
        )
    return points


def validate_and_normalize_config(config: dict) -> dict:
    """Validate grid, contiguous layers, geometry, plots, and forward settings."""
    grid = _mapping(config, "grid")
    nx = _integer(grid, "nx", 11)
    ny = _integer(grid, "ny", 11)
    nz = _integer(grid, "nz", 11)
    dx = _positive_float(grid, "dx")
    dy = _positive_float(grid, "dy")
    dz = _positive_float(grid, "dz")
    storage_dtype = str(config.get("storage_dtype", "float64")).lower()
    if storage_dtype not in {"float32", "float64"}:
        raise ValueError("storage_dtype must be float32 or float64")

    layers = config.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("layers must be a non-empty list")
    normalized_layers = []
    expected_start = 0
    for index, raw_layer in enumerate(layers):
        if not isinstance(raw_layer, dict):
            raise ValueError(f"Layer {index+1} must be a JSON object")
        layer = dict(raw_layer)
        start = _integer(layer, "z_start", 0)
        end = _integer(layer, "z_end", 1)
        if start != expected_start:
            raise ValueError(f"Layer {index+1} has z_start={start}; expected {expected_start} for contiguous layers")
        if end <= start or end > nz:
            raise ValueError(f"Layer {index+1} has invalid range [{start},{end}) for nz={nz}")
        normalized = {
            "name": str(layer.get("name", f"Layer {index+1}")),
            "color": str(layer.get("color", f"C{index % 10}")),
            "z_start": start,
            "z_end": end,
        }
        for name in MATERIAL_FIELDS:
            normalized[name] = _finite_float(layer, name)
        normalized_layers.append(normalized)
        expected_start = end
    if expected_start != nz:
        raise ValueError(f"Layers cover only up to z={expected_start}; full coverage through nz={nz} is required")

    material_columns = {
        name: np.asarray([layer[name] for layer in normalized_layers], dtype=np.float64).reshape(-1, 1, 1)
        for name in MATERIAL_FIELDS
    }
    material_validation = validate_vti_material(
        material_columns["rho"], material_columns["C11"], material_columns["C13"],
        material_columns["C33"], material_columns["C44"], material_columns["C66"],
    )

    pml = _mapping(config, "pml")
    pml_enabled = bool(pml.get("enabled", True))
    nab = _integer(pml, "nAB", 1) if pml_enabled else 0
    if pml_enabled and 2 * nab >= min(nx, ny, nz):
        raise ValueError(f"PML thickness nAB={nab} is too large; require 2*nAB < min(nx,ny,nz)")
    bc_type = str(pml.get("BCtype", "topABC"))
    if bc_type not in {"topABC", "topFS"}:
        raise ValueError("pml.BCtype must be topABC or topFS")
    reflection = _positive_float(pml, "reflection_coefficient")
    if reflection >= 1.0:
        raise ValueError("pml.reflection_coefficient must be in (0,1)")
    polynomial_power = _positive_float(pml, "polynomial_power")

    source = _mapping(config, "source")
    source_xyz = np.asarray([
        _integer(source, "x", 0), _integer(source, "y", 0), _integer(source, "z", 0)
    ], dtype=np.int64)
    if not (source_xyz[0] < nx and source_xyz[1] < ny and source_xyz[2] < nz):
        raise ValueError(f"Source coordinates {source_xyz.tolist()} are out of bounds")
    total_time = _positive_float(source, "total_time")
    dt = _positive_float(source, "dt")
    f0 = _positive_float(source, "f0")
    t0 = _finite_float(source, "t0")
    source_scale = _finite_float(source, "scale")
    if t0 < 0.0:
        raise ValueError("source.t0 must not be negative")

    receivers = parse_receivers(_mapping(config, "receivers"), (nz, ny, nx))
    plot = _mapping(config, "plot")
    line_style = {"outline_width": 1.6, "interface_width": 1.0, **plot}
    plot_values = {
        "enabled": bool(plot.get("enabled", True)),
        "render_style": str(plot.get("render_style", "reference")),
        "filename": str(plot.get("filename", "model_layers_3d.png")),
        "model_filename": str(plot.get("model_filename", "model_layers_only.png")),
        "parameters_filename": str(plot.get("parameters_filename", "model_layer_parameters.png")),
        "dpi": _integer(plot, "dpi", 72),
        "fig_width": _positive_float(plot, "fig_width"),
        "fig_height": _positive_float(plot, "fig_height"),
        "elevation": _finite_float(plot, "elevation"),
        "azimuth": _finite_float(plot, "azimuth"),
        "layer_alpha": _finite_float(plot, "layer_alpha"),
        "outline_color": str(plot.get("outline_color", "#20252b")),
        "outline_width": _positive_float(line_style, "outline_width"),
        "interface_width": _positive_float(line_style, "interface_width"),
        "show_source": bool(plot.get("show_source", True)),
        "show_receivers": bool(plot.get("show_receivers", True)),
        "show_xyz_axes": bool(plot.get("show_xyz_axes", True)),
        "axis_arrow_fraction": _positive_float(plot, "axis_arrow_fraction"),
    }
    if not 0.0 < plot_values["layer_alpha"] <= 1.0:
        raise ValueError("plot.layer_alpha must be in (0,1]")
    if plot_values["render_style"] not in {"reference", "wireframe"}:
        raise ValueError("plot.render_style must be reference or wireframe")
    if not is_color_like(plot_values["outline_color"]):
        raise ValueError("plot.outline_color must be a valid Matplotlib color")
    if plot_values["axis_arrow_fraction"] > 0.5:
        raise ValueError("plot.axis_arrow_fraction must be in (0,0.5]")
    # Old configurations retain their single combined image unless modes are specified.
    output_modes = plot.get("output_modes", ["combined"])
    if (not isinstance(output_modes, list) or not output_modes
            or any(not isinstance(mode, str) or mode not in PLOT_MODES for mode in output_modes)):
        raise ValueError("plot.output_modes must be a non-empty list containing combined, model, or parameters")
    if len(set(output_modes)) != len(output_modes):
        raise ValueError("plot.output_modes must not contain duplicate entries")
    plot_values["output_modes"] = list(output_modes)
    selected_names = []
    for mode in output_modes:
        key = "filename" if mode == "combined" else f"{mode}_filename"
        filename = plot_values[key]
        if (not filename or Path(filename).name != filename
                or filename in {".", ".."} or not Path(filename).suffix):
            raise ValueError(f"plot.{key} must be a filename with an extension and no directory components")
        selected_names.append(filename.casefold())
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("Plot outputs must use distinct filenames")
    if set(selected_names).intersection(name.casefold() for name in CASE_FILENAMES):
        raise ValueError("Plot filenames must not conflict with model data, configuration, or run scripts")

    forward = _mapping(config, "forward")
    forward_values = {
        "python_executable": str(forward.get("python_executable", "python")),
        "source_kind": str(forward.get("source_kind", "dc1")),
        "order": _integer(forward, "order", 2),
        "dtype": str(forward.get("dtype", "fp64")),
        "block": _integer(forward, "block", 1),
        "snapshot_step": _integer(forward, "snapshot_step"),
        "output_dir": str(forward.get("output_dir", "forward_output")),
        "output_prefix": str(forward.get("output_prefix", "layered_vti")),
    }
    if forward_values["source_kind"] not in {"dc1", "dc2", "both"}:
        raise ValueError("forward.source_kind must be dc1, dc2, or both")
    if forward_values["order"] not in {2, 4, 6, 8, 10}:
        raise ValueError("forward.order must be 2, 4, 6, 8, or 10")
    if forward_values["dtype"] not in {"fp32", "fp64"}:
        raise ValueError("forward.dtype must be fp32 or fp64")
    if min(nx, ny, nz) <= forward_values["order"]:
        raise ValueError("Grid dimensions must exceed the finite-difference order")

    return {
        "grid": {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dy, "dz": dz},
        "storage_dtype": storage_dtype,
        "mat_compression": bool(config.get("mat_compression", False)),
        "layers": normalized_layers,
        "material_validation": material_validation,
        "pml": {
            "enabled": pml_enabled,
            "nAB": nab,
            "BCtype": bc_type,
            "reflection_coefficient": reflection,
            "polynomial_power": polynomial_power,
        },
        "source": {
            "xyz": source_xyz,
            "total_time": total_time,
            "dt": dt,
            "f0": f0,
            "t0": t0,
            "scale": source_scale,
        },
        "receivers": receivers,
        "plot": plot_values,
        "forward": forward_values,
    }


def build_material_arrays(normalized: dict) -> dict[str, np.ndarray]:
    """Fill each depth interval with constant material values in [z, x, y] order."""
    grid = normalized["grid"]
    # Preserve FDwave3D MAT axis order; the loader later swaps x and y.
    shape = (grid["nz"], grid["nx"], grid["ny"])
    dtype = np.float32 if normalized["storage_dtype"] == "float32" else np.float64
    fields = {name: np.empty(shape, dtype=dtype) for name in MATERIAL_FIELDS}
    for layer in normalized["layers"]:
        region = slice(layer["z_start"], layer["z_end"])
        for name in MATERIAL_FIELDS:
            fields[name][region, :, :] = layer[name]
    return fields


def build_pml(normalized: dict, vp_vertical: np.ndarray) -> dict[str, np.ndarray]:
    """Build directional damping arrays; topFS leaves the upper z side undamped."""
    grid, pml = normalized["grid"], normalized["pml"]
    nz, nx, ny = grid["nz"], grid["nx"], grid["ny"]
    dtype = vp_vertical.dtype
    pmlx = np.zeros((nz, nx, ny), dtype=dtype)
    pmly = np.zeros_like(pmlx)
    pmlz = np.zeros_like(pmlx)
    if not pml["enabled"]:
        return {"pmlx": pmlx, "pmly": pmly, "pmlz": pmlz}

    nab = pml["nAB"]
    power = pml["polynomial_power"]
    vmax = float(np.max(vp_vertical))
    coefficient = -math.log(pml["reflection_coefficient"]) * (power + 1.0) * vmax
    coefficient /= 2.0 * nab ** (power + 1.0)
    for offset in range(nab):
        distance = float(nab - offset)
        pmlx[:, offset, :] = distance ** power * coefficient / grid["dx"]
        pmlx[:, nx - 1 - offset, :] = distance ** power * coefficient / grid["dx"]
        pmly[:, :, offset] = distance ** power * coefficient / grid["dy"]
        pmly[:, :, ny - 1 - offset] = distance ** power * coefficient / grid["dy"]
        if pml["BCtype"] == "topABC":
            pmlz[offset, :, :] = distance ** power * coefficient / grid["dz"]
        pmlz[nz - 1 - offset, :, :] = distance ** power * coefficient / grid["dz"]
    return {"pmlx": pmlx, "pmly": pmly, "pmlz": pmlz}


def ricker_source(source_config: dict) -> dict[str, np.ndarray | float | int | str]:
    """Generate the Ricker samples and FDwave3D-compatible source metadata."""
    dt, total_time = source_config["dt"], source_config["total_time"]
    count = int(math.floor(total_time / dt + 0.5)) + 1
    time = dt * np.arange(count, dtype=np.float64)
    tau = time - source_config["t0"]
    f0 = source_config["f0"]
    source = source_config["scale"] * (
        1.0 - 2.0 * (np.pi * f0 * tau) ** 2
    ) * np.exp(-(np.pi * f0 * tau) ** 2)
    tdn = int(math.floor(source_config["t0"] / dt + 0.5))
    tdn += 2 * int(math.floor(math.sqrt(6.0) / (np.pi * f0) / dt + 0.5))
    return {
        "src_name": "Ricker", "T": total_time, "dt": dt, "f0": f0,
        "t0": source_config["t0"], "src_scale": source_config["scale"],
        "src": source, "t": time, "N": count, "Ns": tdn, "tdn": tdn,
    }


def matlab_linear_indices(points_xyz: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Convert zero-based [x, y, z] points to one-based [z, x, y] linear indices."""
    nz, _ny, nx = shape_zyx
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    return np.ascontiguousarray(z + x * nz + y * nz * nx + 1, dtype=np.int64)


def geometry_payloads(normalized: dict) -> tuple[dict, dict]:
    """Prepare source and receiver MAT payloads with one-based coordinates."""
    grid = normalized["grid"]
    shape_zyx = (grid["nz"], grid["ny"], grid["nx"])
    source_point = normalized["source"]["xyz"].reshape(1, 3)
    receiver_points = normalized["receivers"]

    source_one_based = source_point + 1
    receivers_one_based = receiver_points + 1
    source_payload = {
        "geometry_src": matlab_linear_indices(source_point, shape_zyx),
        "snx_vec": source_one_based[:, 0], "sny_vec": source_one_based[:, 1],
        "snz_vec": source_one_based[:, 2], "src_n": 1,
        "geometry_src_type": "single",
    }
    receiver_payload = {
        "geometry_rec": matlab_linear_indices(receiver_points, shape_zyx),
        "rnx_vec": receivers_one_based[:, 0], "rny_vec": receivers_one_based[:, 1],
        "rnz_vec": receivers_one_based[:, 2], "rec_n": receiver_points.shape[0],
        "geometry_rec_type": "configured",
    }
    return source_payload, receiver_payload


def _box_faces(x0, x1, y0, y1, z0, z1):
    p000, p100, p110, p010 = (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)
    p001, p101, p111, p011 = (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
    return [
        [p000, p100, p110, p010], [p001, p101, p111, p011],
        [p000, p100, p101, p001], [p100, p110, p111, p101],
        [p110, p010, p011, p111], [p010, p000, p001, p011],
    ]


def _rectangle_edges(xmax: float, ymax: float, z: float) -> list:
    corners = [(0.0, 0.0, z), (xmax, 0.0, z), (xmax, ymax, z), (0.0, ymax, z)]
    return [[corners[index], corners[(index + 1) % 4]] for index in range(4)]


def _draw_model_edges(normalized: dict, axis, xmax: float, ymax: float, zmax: float) -> None:
    """Opaque strokes above translucent fills, below source/receiver markers."""
    plot = normalized["plot"]
    outer_edges = _rectangle_edges(xmax, ymax, 0.0) + _rectangle_edges(xmax, ymax, zmax)
    outer_edges.extend(
        [[(x, y, 0.0), (x, y, zmax)] for x in (0.0, xmax) for y in (0.0, ymax)]
    )
    outline = Line3DCollection(
        outer_edges, colors=plot["outline_color"], linewidths=plot["outline_width"],
        alpha=1.0, zorder=5, label="_nolegend_",
    )
    outline.set_gid("model_outline")
    axis.add_collection3d(outline)

    interface_edges = []
    for layer in normalized["layers"][1:]:
        depth = layer["z_start"] * normalized["grid"]["dz"]
        # A last layer one grid point thick shares the bottom outline.
        if 0.0 < depth < zmax:
            interface_edges.extend(_rectangle_edges(xmax, ymax, depth))
    if interface_edges:
        interfaces = Line3DCollection(
            interface_edges, colors=plot["outline_color"], linewidths=plot["interface_width"],
            alpha=1.0, zorder=4, label="_nolegend_",
        )
        interfaces.set_gid("layer_interfaces")
        axis.add_collection3d(interfaces)


def _draw_model_geometry(normalized: dict, axis) -> None:
    """Shared geometry and marker switches for combined/model-only figures."""
    grid, plot = normalized["grid"], normalized["plot"]
    reference_style = plot["render_style"] == "reference"
    # Reference style uses natural depth sorting, including translucent occlusion.
    # The optional wireframe style instead overlays opaque edges and markers.
    axis.computed_zorder = reference_style
    xmax = (grid["nx"] - 1) * grid["dx"]
    ymax = (grid["ny"] - 1) * grid["dy"]
    zmax = (grid["nz"] - 1) * grid["dz"]
    legend_handles = []
    for layer in normalized["layers"]:
        z0 = layer["z_start"] * grid["dz"]
        z1 = min(layer["z_end"] * grid["dz"], zmax)
        collection = Poly3DCollection(
            _box_faces(0.0, xmax, 0.0, ymax, z0, z1),
            facecolor=layer["color"],
            edgecolor="#30343b" if reference_style else "none",
            linewidth=0.65 if reference_style else 0.0,
            alpha=plot["layer_alpha"],
        )
        axis.add_collection3d(collection)
        label = f"{layer['name']}  [{z0:g}, {z1:g}] m"
        legend_handles.append(Patch(facecolor=layer["color"], edgecolor="#30343b", label=label))

    if not reference_style:
        _draw_model_edges(normalized, axis, xmax, ymax, zmax)

    spacing = np.asarray([grid["dx"], grid["dy"], grid["dz"]], dtype=np.float64)
    if plot["show_source"]:
        source_xyz = normalized["source"]["xyz"] * spacing
        axis.scatter(*source_xyz, marker="*", s=150, color="#d62728", edgecolors="black", linewidths=0.7,
                     depthshade=reference_style, zorder=10, label="Source")
    if plot["show_receivers"]:
        receiver_xyz = normalized["receivers"] * spacing
        axis.scatter(receiver_xyz[:, 0], receiver_xyz[:, 1], receiver_xyz[:, 2], marker="v", s=22,
                     color="#111827", depthshade=False, zorder=11, label="Receivers")

    if plot["show_xyz_axes"]:
        fraction = plot["axis_arrow_fraction"]
        arrow_lengths = (xmax * fraction, ymax * fraction, zmax * fraction)
        arrow_specs = (
            (arrow_lengths[0], 0.0, 0.0, "#d62728", "X"),
            (0.0, arrow_lengths[1], 0.0, "#2ca02c", "Y"),
            (0.0, 0.0, arrow_lengths[2], "#1f77b4", "Z"),
        )
        for vx, vy, vz, color, label in arrow_specs:
            axis.quiver(
                0.0, 0.0, 0.0, vx, vy, vz, color=color, linewidth=2.4,
                arrow_length_ratio=0.12, normalize=False, zorder=20,
            )
            axis.text(
                vx * 1.12, vy * 1.12, vz * 1.12, label, color=color,
                fontsize=12, fontweight="bold", ha="center", va="center",
                zorder=3 if reference_style else 21,
            )

    axis.set_xlim(0.0, xmax)
    axis.set_ylim(0.0, ymax)
    axis.set_zlim(zmax, 0.0)
    if not reference_style:
        for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate_axis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    axis.set_xlabel("X (m)", labelpad=9)
    axis.set_ylabel("Y (m)", labelpad=9)
    axis.set_zlabel("Z / depth (m)", labelpad=9)
    axis.view_init(elev=plot["elevation"], azim=plot["azimuth"])
    try:
        axis.set_box_aspect((max(xmax, 1.0), max(ymax, 1.0), max(zmax, 1.0)))
    except AttributeError:
        pass
    handles, _labels = axis.get_legend_handles_labels()
    if reference_style:
        axis.set_title("Layered VTI model geometry")
        axis.legend(handles=legend_handles + handles, loc="upper left",
                    bbox_to_anchor=(0.0, 0.98), frameon=True)
    else:
        axis.legend(handles=legend_handles + handles, loc="lower center",
                    bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8, frameon=True)


def _draw_layer_parameters(normalized: dict, table_axis) -> None:
    """Draw the same layer values and units in combined/parameters-only figures."""
    grid = normalized["grid"]
    zmax = (grid["nz"] - 1) * grid["dz"]
    table_axis.axis("off")
    table_rows = []
    for layer in normalized["layers"]:
        z0 = layer["z_start"] * grid["dz"]
        z1 = min(layer["z_end"] * grid["dz"], zmax)
        table_rows.append([
            layer["name"], f"{z0:g}–{z1:g}", f"{layer['rho']:g}",
            *[f"{layer[name] / 1.0e9:.3g}" for name in ("C11", "C13", "C33", "C44", "C66")],
        ])
    columns = ["Layer", "z (m)", "rho", "C11", "C13", "C33", "C44", "C66"]
    table = table_axis.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#f9fafb" if row % 2 else "#ffffff")
    table_axis.set_title("Layer parameters (rho: kg/m³; stiffness: GPa)", pad=18)


def create_layered_model_figure(normalized: dict, mode: str = "combined"):
    """Create one figure; the caller owns it and must close it after use."""
    if mode not in PLOT_MODES:
        raise ValueError(f"Unknown plot mode {mode!r}; choose from {PLOT_MODES}")
    grid, plot = normalized["grid"], normalized["plot"]
    width, height = plot["fig_width"], plot["fig_height"]
    if mode == "model":
        width = min(width, height)
    elif mode == "parameters":
        height = max(3.0, 1.8 + 0.4 * (len(normalized["layers"]) + 1))
    figure = plt.figure(figsize=(width, height), constrained_layout=True)
    if mode == "combined":
        _draw_model_geometry(normalized, figure.add_subplot(1, 2, 1, projection="3d"))
        _draw_layer_parameters(normalized, figure.add_subplot(1, 2, 2))
    elif mode == "model":
        _draw_model_geometry(normalized, figure.add_subplot(1, 1, 1, projection="3d"))
    else:
        _draw_layer_parameters(normalized, figure.add_subplot(1, 1, 1))
    figure.suptitle(
        f"VTI model | grid={grid['nx']}×{grid['ny']}×{grid['nz']} | "
        f"spacing=({grid['dx']:g}, {grid['dy']:g}, {grid['dz']:g}) m",
        fontsize=14,
    )
    return figure


def plot_layered_model(normalized: dict, output: Path, mode: str = "combined") -> None:
    """Save the selected model view and close its figure even if saving fails."""
    figure = create_layered_model_figure(normalized, mode)
    try:
        figure.savefig(output, dpi=normalized["plot"]["dpi"], bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def selected_plot_paths(normalized: dict, output_dir: Path) -> dict[str, Path]:
    """Map enabled plot modes to their configured output paths."""
    plot = normalized["plot"]
    if not plot["enabled"]:
        return {}
    return {
        mode: output_dir / plot["filename" if mode == "combined" else f"{mode}_filename"]
        for mode in plot["output_modes"]
    }


def write_run_script(normalized: dict, output_dir: Path) -> Path:
    """Write a PowerShell launcher with the configured interpreter and solver flags."""
    forward = normalized["forward"]
    main_script = Path(__file__).resolve().parent / "main.py"
    output_path = output_dir / forward["output_dir"]
    command = (
        f'& "{forward["python_executable"]}" -B "{main_script}" `\n'
        f'  --case-dir "$PSScriptRoot" `\n'
        f'  --source {forward["source_kind"]} `\n'
        f'  --order {forward["order"]} `\n'
        f'  --dtype {forward["dtype"]} `\n'
        f'  --block {forward["block"]} `\n'
        f'  --snapshot-step {forward["snapshot_step"]} `\n'
        f'  --output-prefix "{forward["output_prefix"]}" `\n'
        f'  --output-dir "{output_path}"\n'
    )
    path = output_dir / "run_forward.ps1"
    path.write_text(command, encoding="utf-8-sig")
    return path


def build_case(config: dict, config_path: Path, output_dir: Path, overwrite=False) -> dict:
    """Write model, source, boundary, geometry, plots, launcher, and metadata files."""
    normalized = validate_and_normalize_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in CASE_FILENAMES if (output_dir / name).exists()]
    plot_paths = selected_plot_paths(normalized, output_dir)
    existing.extend(path.name for path in plot_paths.values() if path.exists())
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}. Add --overwrite to replace them."
        )

    fields = build_material_arrays(normalized)
    rho = fields["rho"]
    vp_vertical = np.ascontiguousarray(np.sqrt(fields["C33"] / rho), dtype=rho.dtype)
    vs_vertical = np.ascontiguousarray(np.sqrt(fields["C44"] / rho), dtype=rho.dtype)
    pml_arrays = build_pml(normalized, vp_vertical)
    source_payload = ricker_source(normalized["source"])
    source_geometry, receiver_geometry = geometry_payloads(normalized)
    grid, pml = normalized["grid"], normalized["pml"]
    compression = normalized["mat_compression"]

    model_payload = {
        **grid, "rhom": rho, "vpm": vp_vertical, "vsm": vs_vertical,
        "wave_type": "elastic", "m_name": "layered_VTI_10",
        **{name: fields[name] for name in ("C11", "C13", "C33", "C44", "C66")},
    }
    savemat(output_dir / "model.mat", model_payload, do_compression=compression, oned_as="row")
    savemat(output_dir / "source.mat", source_payload, do_compression=compression, oned_as="row")
    savemat(
        output_dir / "BC.mat",
        {"BC": pml_arrays, "nAB": pml["nAB"], "BCtype": pml["BCtype"],
         "BCname": "PML" if pml["enabled"] else "NONE"},
        do_compression=compression, oned_as="row",
    )
    savemat(output_dir / "geometry_src.mat", source_geometry, do_compression=compression, oned_as="row")
    savemat(output_dir / "geometry_rec.mat", receiver_geometry, do_compression=compression, oned_as="row")

    for mode, plot_path in plot_paths.items():
        plot_layered_model(normalized, plot_path, mode)
    used_config_path = output_dir / "model_config_used.json"
    used_config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    run_script_path = write_run_script(normalized, output_dir)

    vmax = float(np.max(vp_vertical))
    # Report a vertical-speed indicator, without treating it as a CFL validation.
    courant_indicator = normalized["source"]["dt"] * vmax * math.sqrt(
        1.0 / grid["dx"] ** 2 + 1.0 / grid["dy"] ** 2 + 1.0 / grid["dz"] ** 2
    )
    summary = {
        "case_dir": str(output_dir),
        "config_source": str(config_path),
        "grid": grid,
        "matlab_array_shape_zxy": [grid["nz"], grid["nx"], grid["ny"]],
        "layer_count": len(normalized["layers"]),
        "layers": normalized["layers"],
        "source_xyz_zero_based": normalized["source"]["xyz"].tolist(),
        "receiver_count": int(normalized["receivers"].shape[0]),
        "pml": pml,
        "vertical_vp_range_m_per_s": [float(np.min(vp_vertical)), vmax],
        "vertical_vs_range_m_per_s": [float(np.min(vs_vertical)), float(np.max(vs_vertical))],
        "courant_indicator_diagnostic": courant_indicator,
        "vti_validation": normalized["material_validation"],
        "files": {
            "model": str(output_dir / "model.mat"), "source": str(output_dir / "source.mat"),
            "boundary": str(output_dir / "BC.mat"), "source_geometry": str(output_dir / "geometry_src.mat"),
            "receiver_geometry": str(output_dir / "geometry_rec.mat"),
            # Keep the legacy combined-image key; list all selected images separately.
            "plot": str(plot_paths["combined"]) if "combined" in plot_paths else None,
            "plots": {mode: str(path) for mode, path in plot_paths.items()},
            "run_script": str(run_script_path),
        },
    }
    summary_path = output_dir / "model_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    """Parse builder arguments and print the generated case summary."""
    default_config = Path(__file__).resolve().parent / "layered_vti_config.json"
    parser = argparse.ArgumentParser(
        description="Build a fully configurable multilayer VTI case for version 10"
    )
    parser.add_argument("--config", default=str(default_config), help="JSON configuration path")
    parser.add_argument("--output-dir", help="Override output_case_dir in the JSON configuration")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated case files")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    output_dir = resolve_output_dir(config, config_path, args.output_dir)
    summary = build_case(config, config_path, output_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

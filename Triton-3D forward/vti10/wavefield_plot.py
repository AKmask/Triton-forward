"""Render source-centred XY, XZ, and YZ wavefield slices and a 3-D view.

Match compare_wavefield_planes.py with a shared symmetric color scale,
nearest-neighbour cells, and thin gray axes. The 3-D layout keeps transparent
backgrounds and geometrically clipped rectangle outlines. Solver diagnostics are never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


PLANE_COMPONENTS = ("vx", "vy", "vz")
PLOT_COMPONENTS = ("magnitude", *PLANE_COMPONENTS)


def extract_wavefield_planes(
    result: Mapping,
    component: str = "vx",
) -> dict[str, np.ndarray]:
    """Return XY/XZ/YZ planes for one component or the velocity magnitude."""
    if component not in PLOT_COMPONENTS:
        raise ValueError(f"plot component must be one of {PLOT_COMPONENTS}, got {component!r}")

    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise KeyError("forward result does not contain diagnostics")

    planes: dict[str, np.ndarray] = {}
    for plane in ("xy", "xz", "yz"):
        arrays = []
        for velocity_component in PLANE_COMPONENTS:
            key = f"snapshot_{velocity_component}_{plane}"
            if key not in diagnostics:
                raise KeyError(f"forward diagnostics is missing {key}")
            array = np.asarray(diagnostics[key], dtype=np.float64)
            if array.ndim != 2:
                raise ValueError(f"{key} must be two-dimensional, got shape={array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{key} contains NaN or Inf")
            arrays.append(array)

        if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
            raise ValueError(f"the three velocity components on the {plane.upper()} plane have different shapes")
        if component == "magnitude":
            planes[plane] = np.hypot(np.hypot(arrays[0], arrays[1]), arrays[2])
        else:
            planes[plane] = arrays[PLANE_COMPONENTS.index(component)]
    return planes


def _expected_shapes(case) -> dict[str, tuple[int, int]]:
    return {
        "xy": (int(case.ny), int(case.nx)),
        "xz": (int(case.nz), int(case.nx)),
        "yz": (int(case.nz), int(case.ny)),
    }


def _common_norm(planes: Mapping[str, np.ndarray], component: str) -> Normalize:
    maximum = max(float(np.max(np.abs(array))) for array in planes.values())
    if not np.isfinite(maximum) or maximum == 0.0:
        maximum = 1.0
    # Use the reference script's symmetric scale, including magnitude plots.
    return Normalize(vmin=-maximum, vmax=maximum)


def _component_label(component: str) -> str:
    quantity = "Velocity magnitude" if component == "magnitude" else component.upper()
    return f"{quantity} (simulation units)"


def _plane_extent(case, plane: str) -> tuple[float, float, float, float]:
    """Place pixels around grid nodes, exactly as in the comparison script."""
    shape = _expected_shapes(case)[plane]
    horizontal_spacing, vertical_spacing = {
        "xy": (float(case.dx), float(case.dy)),
        "xz": (float(case.dx), float(case.dz)),
        "yz": (float(case.dy), float(case.dz)),
    }[plane]
    return (
        -horizontal_spacing / 2,
        (shape[1] - 0.5) * horizontal_spacing,
        (shape[0] - 0.5) * vertical_spacing,
        -vertical_spacing / 2,
    )


def _style_colorbar(colorbar, component: str) -> None:
    """Apply the reference figure's scientific notation and fine gray outline."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    colorbar.formatter = formatter
    colorbar.update_ticks()
    colorbar.set_label(_component_label(component), fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.outline.set_linewidth(0.6)
    colorbar.outline.set_edgecolor("#6b7280")


def _plot_single_plane(
    array: np.ndarray,
    plane: str,
    case,
    component: str,
    step: int,
    time_s: float,
    cmap,
    norm: Normalize,
    output_path: Path,
    dpi: int,
) -> None:
    axis_names = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}
    fig, ax = plt.subplots(figsize=(8.0, 6.4), constrained_layout=True)
    image = ax.imshow(
        array,
        cmap=cmap,
        norm=norm,
        extent=_plane_extent(case, plane),
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )
    horizontal, vertical = axis_names[plane]
    fixed_axis = {"xy": "Z", "xz": "Y", "yz": "X"}[plane]
    fixed_index = {"xy": case.src_z, "xz": case.src_y, "yz": case.src_x}[plane]
    fixed_spacing = {"xy": case.dz, "xz": case.dy, "yz": case.dx}[plane]
    ax.set_xlabel(f"{horizontal} (m)", fontsize=10)
    ax.set_ylabel(f"{vertical} (m)", fontsize=10)
    ax.set_title(
        f"{plane.upper()} wavefield at {fixed_axis}={int(fixed_index) * float(fixed_spacing):g} m\n"
        f"{component.upper()}, step={step}, time={time_s:.6g} s",
        fontsize=11, pad=10,
    )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#6b7280")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.025, aspect=35)
    _style_colorbar(colorbar, component)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _sample_indices(size: int, maximum_points: int) -> np.ndarray:
    if maximum_points == 0 or size <= maximum_points:
        return np.arange(size, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, size - 1, maximum_points)).astype(np.int64))


def _sample_edges(indices: np.ndarray, size: int, spacing: float) -> np.ndarray:
    """Give each retained grid value a nearest-neighbour cell centred on its node.

    With all nodes retained, boundaries are exactly half a grid spacing away.
    An explicit point cap uses nearest-neighbour cells without amplitude averaging.
    """
    centres = indices * spacing
    return np.concatenate(([-0.5 * spacing], 0.5 * (centres[:-1] + centres[1:]),
                           [(size - 0.5) * spacing]))


def _plane_faces(x, y, z, values, cmap, norm, alpha):
    """Color one face per original sample, without averaging adjacent amplitudes."""
    points = np.stack((x, y, z), axis=-1)
    vertices = np.stack(
        (points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]),
        axis=2,
    ).reshape(-1, 4, 3)
    cell_values = values.ravel()
    colors = np.asarray(cmap(norm(cell_values))).copy()
    scale = max(abs(float(norm.vmin)), abs(float(norm.vmax)), np.finfo(float).tiny)
    # Transparency is specific to the intersecting 3-D view: it prevents white
    # backgrounds from hiding other slices. RGB colors use the unaltered samples.
    colors[:, 3] = alpha * np.clip(np.abs(cell_values) / (0.15 * scale), 0.0, 1.0)
    visible = colors[:, 3] > 0.002
    return vertices[visible], colors[visible]


def _slice_rectangles(bounds: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the four perimeter vertices of the XY, XZ, and YZ slices."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    sx, sy, sz = source
    return (
        np.array([(xmin, ymin, sz), (xmax, ymin, sz),
                  (xmax, ymax, sz), (xmin, ymax, sz)]),
        np.array([(xmin, sy, zmin), (xmax, sy, zmin),
                  (xmax, sy, zmax), (xmin, sy, zmax)]),
        np.array([(sx, ymin, zmin), (sx, ymax, zmin),
                  (sx, ymax, zmax), (sx, ymin, zmax)]),
    )


def _hidden_edge_interval(start, end, rectangle, normal_axis, toward_camera):
    """Clip an edge parameter t in [0, 1] against a rectangle's viewing shadow.

    A point is hidden only if its ray toward the camera meets the finite slice
    in front of it. Ray distance and hit coordinates are affine in t, so their
    bounds give an exact interval without sampling the line or the image.
    """
    if abs(toward_camera[normal_axis]) < 1e-12:
        return None  # An edge-on slice has no projected area to occlude a line.
    lower = rectangle.min(axis=0)
    upper = rectangle.max(axis=0)
    tolerance = max(float(np.max(upper - lower)), 1.0) * 1e-10
    delta = end - start
    distance = (lower[normal_axis] - start[normal_axis]) / toward_camera[normal_axis]
    distance_delta = -delta[normal_axis] / toward_camera[normal_axis]
    hit = start + distance * toward_camera
    hit_delta = delta + distance_delta * toward_camera
    constraints = [(distance, distance_delta, tolerance, np.inf)]
    for axis in range(3):
        if axis != normal_axis:
            # Include grazing rays on the perimeter despite projection roundoff.
            constraints.append((hit[axis], hit_delta[axis],
                                lower[axis] - tolerance, upper[axis] + tolerance))
    t0, t1 = 0.0, 1.0
    for value, slope, minimum, maximum in constraints:
        if abs(slope) <= tolerance:
            if value < minimum or value > maximum:
                return None
            continue
        a, b = (minimum - value) / slope, (maximum - value) / slope
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t1 <= t0:
            return None
    return t0, t1


def _visible_slice_edges(rectangles, projection: np.ndarray) -> list[np.ndarray]:
    """Keep only visible portions of the twelve original rectangle edges.

    The inverse orthographic projection supplies the viewing direction in data
    coordinates, including reversed depth and unequal physical axis lengths.
    Slice backgrounds occlude outline lines even when wavefield colors fade
    to transparency. No lines are introduced along the slice intersections.
    """
    direction = np.linalg.solve(projection, np.array([0.0, 0.0, -1.0, 0.0]))[:3]
    direction /= np.linalg.norm(direction)
    segments = []
    for owner, rectangle in enumerate(rectangles):
        for start, end in zip(rectangle, np.roll(rectangle, -1, axis=0)):
            visible = [(0.0, 1.0)]
            for other, (occluder, normal_axis) in enumerate(zip(rectangles, (2, 1, 0))):
                if owner == other:
                    continue
                hidden = _hidden_edge_interval(start, end, occluder, normal_axis, direction)
                if hidden is None:
                    continue
                h0, h1 = hidden
                remaining = []
                for t0, t1 in visible:
                    if h1 <= t0 or h0 >= t1:
                        remaining.append((t0, t1))
                    else:
                        if h0 > t0:
                            remaining.append((t0, h0))
                        if h1 < t1:
                            remaining.append((h1, t1))
                visible = remaining
            for t0, t1 in visible:
                if t1 - t0 > 1e-9:
                    segments.append(start + np.array([[t0], [t1]]) * (end - start))
    return segments


def _distance_ticks(maximum: float, include_zero: bool = False) -> np.ndarray:
    """Choose a few readable physical-distance ticks, without exceeding the grid."""
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0.0, maximum)
    lower = ticks >= 0.0 if include_zero else ticks > 0.0
    return ticks[lower & (ticks <= maximum)]


def _plot_orthogonal_3d(
    planes: Mapping[str, np.ndarray],
    case,
    component: str,
    step: int,
    time_s: float,
    cmap,
    norm: Normalize,
    output_path: Path,
    dpi: int,
    alpha: float,
    maximum_points: int,
    elevation: float = 20.0,
    azimuth: float = -60.0,
    show_colorbar: bool = False,
) -> None:
    """Save a clean orthographic view of three outlined, intersecting slices."""
    x_indices = _sample_indices(int(case.nx), maximum_points)
    y_indices = _sample_indices(int(case.ny), maximum_points)
    z_indices = _sample_indices(int(case.nz), maximum_points)
    x = _sample_edges(x_indices, int(case.nx), float(case.dx))
    y = _sample_edges(y_indices, int(case.ny), float(case.dy))
    z = _sample_edges(z_indices, int(case.nz), float(case.dz))
    xy = planes["xy"][np.ix_(y_indices, x_indices)]
    xz = planes["xz"][np.ix_(z_indices, x_indices)]
    yz = planes["yz"][np.ix_(z_indices, y_indices)]

    x_xy, y_xy = np.meshgrid(x, y, indexing="xy")
    x_xz, z_xz = np.meshgrid(x, z, indexing="xy")
    y_yz, z_yz = np.meshgrid(y, z, indexing="xy")
    source_x = int(case.src_x) * float(case.dx)
    source_y = int(case.src_y) * float(case.dy)
    source_z = int(case.src_z) * float(case.dz)
    xmin, xmax = float(x[0]), float(x[-1])
    ymin, ymax = float(y[0]), float(y[-1])
    zmin, zmax = float(z[0]), float(z[-1])
    grids = (
        (x_xy, y_xy, np.full_like(x_xy, source_z), xy),
        (x_xz, np.full_like(x_xz, source_y), z_xz, xz),
        (np.full_like(y_yz, source_x), y_yz, z_yz, yz),
    )
    faces = [_plane_faces(*grid, cmap, norm, alpha) for grid in grids]
    vertices = np.concatenate([item[0] for item in faces], axis=0)
    colors = np.concatenate([item[1] for item in faces], axis=0)

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10,
        "axes.labelsize": 10,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(6.0, 6.0), facecolor="white")
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        try:
            # Sort all slice cells together, rather than sorting whole planes.
            # This avoids a front plane incorrectly obscuring an entire rear one.
            if len(vertices):
                collection = Poly3DCollection(
                    vertices, facecolors=colors, edgecolors="none", linewidths=0,
                    antialiased=False, zsort="average", zorder=2,
                )
                ax.add_collection3d(collection)

            ax.set_proj_type("ortho")
            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmax, zmin)
            ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
            ax.set_xticks(_distance_ticks(xmax))
            ax.set_yticks(_distance_ticks(ymax))
            ax.set_zticks(_distance_ticks(zmax, include_zero=True))
            ax.set_xlabel("X (m)", labelpad=7)
            ax.set_ylabel("Y (m)", labelpad=7)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel("Z (m)", labelpad=9, rotation=-90)
            ax.tick_params(axis="both", labelsize=9, pad=0, colors="#111827")
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_visible(False)
                axis.line.set_color("#6b7280")
                axis.line.set_linewidth(0.6)

            # Draw complete rectangular perimeters after removing their hidden parts.
            # Only visible edges are overlaid; there are no fixed L-shaped frames.
            bounds = np.array([(xmin, xmax), (ymin, ymax), (zmin, zmax)])
            source = np.array([source_x, source_y, source_z])
            rectangles = _slice_rectangles(bounds, source)
            segments = _visible_slice_edges(rectangles, ax.get_proj())
            if segments:
                outlines = Line3DCollection(
                    segments, colors="#6b7280", linewidths=0.6, alpha=0.45,
                    antialiased=True, zorder=3,
                )
                outlines.set_capstyle("butt")
                ax.add_collection3d(outlines)

            # Keep the requested clean 3-D layout; an amplitude bar is optional.
            # Component, step, and time are retained in the PNG Description metadata.
            if show_colorbar:
                scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
                scalar_mappable.set_array([])
                colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.60, pad=0.08, aspect=35)
                _style_colorbar(colorbar, component)
            fig.subplots_adjust(left=0.0, right=0.88, bottom=0.04, top=0.98)
            fig.savefig(
                output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.10, facecolor="white",
                bbox_extra_artists=(ax.xaxis.label, ax.yaxis.label, ax.zaxis.label),
                metadata={"Description": f"{component}; step={step}; time={time_s:.9g} s"},
            )
        finally:
            plt.close(fig)


def save_wavefield_figures(
    result: Mapping,
    case,
    output_dir: str | Path,
    stem: str,
    component: str = "vx",
    cmap_name: str = "seismic",
    dpi: int = 300,
    alpha: float = 0.95,
    maximum_points: int = 0,
    elevation: float = 20.0,
    azimuth: float = -60.0,
    show_colorbar: bool = False,
) -> dict[str, Path]:
    """Save three source-plane figures and their combined 3-D orthogonal view."""
    if not np.isfinite(elevation) or not -90.0 <= elevation <= 90.0:
        raise ValueError("plot elevation must be finite and in [-90,90]")
    if not np.isfinite(azimuth):
        raise ValueError("plot azimuth must be finite")
    if dpi <= 0:
        raise ValueError("plot dpi must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("3-D plane alpha must be in (0, 1]")
    if maximum_points != 0 and maximum_points < 8:
        raise ValueError("plot maximum_points must be 0 (full grid) or at least 8")
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError as exc:
        raise ValueError(f"unknown Matplotlib colormap: {cmap_name}") from exc

    planes = extract_wavefield_planes(result, component)
    for plane, expected_shape in _expected_shapes(case).items():
        if planes[plane].shape != expected_shape:
            raise ValueError(
                f"{plane.upper()} snapshot shape must be {expected_shape}, got {planes[plane].shape}"
            )

    diagnostics = result["diagnostics"]
    step = int(diagnostics["snapshot_step"])
    time_s = float(diagnostics["snapshot_time"])
    norm = _common_norm(planes, component)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        plane: output / f"{stem}_wavefield_{plane}.png"
        for plane in ("xy", "xz", "yz")
    }
    paths["3d"] = output / f"{stem}_wavefield_3d.png"

    for plane in ("xy", "xz", "yz"):
        _plot_single_plane(
            planes[plane],
            plane,
            case,
            component,
            step,
            time_s,
            cmap,
            norm,
            paths[plane],
            dpi,
        )
    _plot_orthogonal_3d(
        planes,
        case,
        component,
        step,
        time_s,
        cmap,
        norm,
        paths["3d"],
        dpi,
        alpha,
        maximum_points,
        elevation=elevation,
        azimuth=azimuth,
        show_colorbar=show_colorbar,
    )
    return paths

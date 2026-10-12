"""Build a 2-D VTI case from Hess MAT arrays and a JSON configuration.

The builder exports model, source, split-PML, and geometry files, plus
PowerShell launchers. Configuration coordinates are zero-based [x, z].
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vti2d.runtime_paths import configure_runtime_cache

# Configure caches before third-party libraries initialize their runtimes.
CACHE_ROOT = configure_runtime_cache()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from vti2d.fd_coeff import conservative_cfl_limit
from vti2d.hess_model import load_hess_material
from vti2d.source import SOURCE_KINDS, ricker
from vti2d.vti_stiffness import INDEPENDENT_VTI_FIELDS, validate_vti_material


GENERATED_FILES = (
    "model.mat",
    "source.mat",
    "BC.mat",
    "geometry_src.mat",
    "geometry_rec.mat",
    "hess_vti2d_model.png",
    "model_config_used.json",
    "model_summary.json",
    "run_gpu.ps1",
    "run_cpu.ps1",
)


def _mapping(parent: dict, name: str) -> dict:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration field {name} must be an object")
    return value


def _finite(parent: dict, name: str) -> float:
    if name not in parent:
        raise ValueError(f"configuration is missing {name}")
    value = float(parent[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(parent: dict, name: str) -> float:
    value = _finite(parent, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _integer(parent: dict, name: str, minimum: int | None = None) -> int:
    if name not in parent:
        raise ValueError(f"configuration is missing {name}")
    raw = parent[name]
    value = int(raw)
    if isinstance(raw, float) and raw != value:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def load_config(path: str | Path) -> tuple[dict, Path]:
    """Read a JSON object and return it with its resolved file path."""
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be an object")
    return config, config_path


def resolve_output_dir(config: dict, config_path: Path, override: str | None) -> Path:
    """Resolve relative output paths against the configuration directory."""
    value = override if override is not None else config.get("output_case_dir")
    if not value:
        raise ValueError("set output_case_dir or use --output-dir")
    output = Path(str(value)).expanduser()
    if not output.is_absolute():
        output = config_path.parent / output
    return output.resolve()


def _parse_receivers(config: dict, nx: int, nz: int) -> np.ndarray:
    """Return validated zero-based [x, z] receiver coordinates."""
    mode = str(config.get("mode", "line")).lower()
    if mode == "line":
        start = _integer(config, "x_start", 0)
        stop = _integer(config, "x_stop", 0)
        step = _integer(config, "x_step", 1)
        z = _integer(config, "z", 0)
        if start > stop:
            raise ValueError("receiver x_start must not exceed x_stop")
        x = np.arange(start, stop + 1, step, dtype=np.int64)
        points = np.column_stack((x, np.full_like(x, z)))
    elif mode == "explicit":
        points = np.asarray(config.get("points_xz", []), dtype=np.int64)
        if points.ndim != 2 or points.shape[1:] != (2,) or points.shape[0] == 0:
            raise ValueError("explicit receivers require non-empty points_xz=[[x,z],...]")
    else:
        raise ValueError("receiver mode must be line or explicit")
    valid = (points[:, 0] >= 0) & (points[:, 0] < nx) & (points[:, 1] >= 0) & (points[:, 1] < nz)
    if not np.all(valid):
        index = int(np.flatnonzero(~valid)[0])
        raise ValueError(f"receiver {index} at {points[index].tolist()} is outside nx/nz={nx}/{nz}")
    return np.ascontiguousarray(points)


def validate_config(config: dict, config_path: str | Path) -> tuple[dict, dict]:
    """Load material fields, validate geometry and CFL, and normalize options."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    hess = load_hess_material(_mapping(config, "hess_model"), resolved_config_path.parent)
    material = hess["material"]
    nz, nx = material["rho"].shape
    grid = _mapping(config, "grid")
    base_dx, base_dz = _positive(grid, "dx"), _positive(grid, "dz")
    stride_z, stride_x = hess["transform"]["stride_zx"]
    # Spatial decimation changes physical spacing as well as array dimensions.
    dx, dz = base_dx * stride_x, base_dz * stride_z
    validation = validate_vti_material(
        material["rho"], material["C11"], material["C13"],
        material["C33"], material["C44"], material["C66"],
    )

    pml = _mapping(config, "pml")
    nab = _integer(pml, "nAB", 1)
    if 2 * nab >= min(nx, nz):
        raise ValueError("PML requires 2*nAB < min(nx,nz)")
    reflection = _positive(pml, "reflection_coefficient")
    power = _positive(pml, "polynomial_power")
    if reflection >= 1.0:
        raise ValueError("reflection_coefficient must be in (0,1)")
    if str(pml.get("BCtype", "topABC")) != "topABC":
        raise ValueError("this accelerated version supports BCtype=topABC only")

    source = _mapping(config, "source")
    source_x, source_z = _integer(source, "x", 0), _integer(source, "z", 0)
    if not (0 <= source_x < nx and 0 <= source_z < nz):
        raise ValueError(f"source [x,z]=[{source_x},{source_z}] is outside nx/nz=[{nx},{nz}]")
    source_kind = str(source.get("kind", "explosive")).lower()
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source.kind must be one of {SOURCE_KINDS}")
    total_time = _positive(source, "total_time")
    dt = _positive(source, "dt")
    f0 = _positive(source, "f0")
    t0 = _finite(source, "t0")
    scale = _finite(source, "scale")
    if t0 < 0.0:
        raise ValueError("source.t0 must not be negative")
    receivers = _parse_receivers(_mapping(config, "receivers"), nx, nz)

    forward = _mapping(config, "forward")
    order = _integer(forward, "order", 2)
    if order not in {2, 4, 6, 8, 10}:
        raise ValueError("forward.order must be 2,4,6,8,10")
    if min(nx, nz) <= order:
        raise ValueError("the transformed Hess dimensions must exceed forward.order")
    half = order // 2
    if not (half <= source_x < nx - half and half <= source_z < nz - half):
        raise ValueError(f"source lies outside the active order-{order} finite-difference region")
    dtype = str(forward.get("dtype", "fp32")).lower()
    if dtype not in {"fp32", "fp64"}:
        raise ValueError("forward.dtype must be fp32 or fp64")
    block = _integer(forward, "block", 1)
    if block not in {64, 128, 256, 512, 1024}:
        raise ValueError("forward.block must be 64,128,256,512,1024")
    record_stride = _integer(forward, "record_stride", 1)
    snapshot_step = _integer(forward, "snapshot_step")
    cfl_limit = conservative_cfl_limit(dx, dz, validation["conservative_vmax_m_s"], order)
    if dt > cfl_limit:
        raise ValueError(f"source.dt={dt:.8e} exceeds conservative CFL limit {cfl_limit:.8e}")
    validation["cfl_dt_limit_s"] = cfl_limit

    plot = _mapping(config, "plot")
    normalized = {
        "grid": {"nx": nx, "nz": nz, "dx": dx, "dz": dz, "input_dx": base_dx, "input_dz": base_dz},
        "mat_compression": bool(config.get("mat_compression", False)),
        "hess": hess,
        "validation": validation,
        "pml": {
            "nAB": nab, "reflection_coefficient": reflection,
            "polynomial_power": power, "BCtype": "topABC",
        },
        "source": {
            "x": source_x, "z": source_z, "kind": source_kind,
            "total_time": total_time, "dt": dt, "f0": f0, "t0": t0, "scale": scale,
        },
        "receivers": receivers,
        "plot": {
            "enabled": bool(plot.get("enabled", True)),
            "filename": str(plot.get("filename", "hess_vti2d_model.png")),
            "cmap": str(plot.get("cmap", "viridis")),
            "anisotropy_cmap": str(plot.get("anisotropy_cmap", "coolwarm")),
            "dpi": _integer(plot, "dpi", 72),
            "fig_width": _positive(plot, "fig_width"),
            "fig_height": _positive(plot, "fig_height"),
        },
        "forward": {
            "python_executable": str(forward.get("python_executable", "python")),
            "order": order, "dtype": dtype, "block": block,
            "record_stride": record_stride, "snapshot_step": snapshot_step,
            "output_dir": str(forward.get("output_dir", "forward_output")),
            "output_prefix": str(forward.get("output_prefix", "hess_vti2d")),
        },
    }
    return normalized, material


def build_pml(normalized: dict, dtype) -> dict[str, np.ndarray]:
    """Construct polynomial x/z damping profiles in [z, x] order."""
    grid, pml = normalized["grid"], normalized["pml"]
    nx, nz = grid["nx"], grid["nz"]
    pmlx = np.zeros((nz, nx), dtype=dtype)
    pmlz = np.zeros_like(pmlx)
    nab, power = pml["nAB"], pml["polynomial_power"]
    vmax = normalized["validation"]["conservative_vmax_m_s"]
    coefficient = -math.log(pml["reflection_coefficient"]) * (power + 1.0) * vmax
    coefficient /= 2.0 * nab ** (power + 1.0)
    for offset in range(nab):
        distance = float(nab - offset)
        pmlx[:, offset] = distance**power * coefficient / grid["dx"]
        pmlx[:, nx - 1 - offset] = distance**power * coefficient / grid["dx"]
        pmlz[offset, :] = distance**power * coefficient / grid["dz"]
        pmlz[nz - 1 - offset, :] = distance**power * coefficient / grid["dz"]
    return {"pmlx": pmlx, "pmlz": pmlz}


def geometry_payloads(normalized: dict) -> tuple[dict, dict]:
    """Convert zero-based geometry to MATLAB coordinates and linear indices."""
    grid, source = normalized["grid"], normalized["source"]
    # MATLAB linear indices are one-based and traverse z before x.
    source_index = source["z"] + source["x"] * grid["nz"] + 1
    receiver_points = normalized["receivers"]
    receiver_indices = receiver_points[:, 1] + receiver_points[:, 0] * grid["nz"] + 1
    return (
        {
            "geometry_src": np.asarray([source_index], dtype=np.int64),
            "snx_vec": np.asarray([source["x"] + 1], dtype=np.int64),
            "snz_vec": np.asarray([source["z"] + 1], dtype=np.int64),
            "src_n": 1,
            "geometry_src_type": "single_2d_hess",
        },
        {
            "geometry_rec": np.asarray(receiver_indices, dtype=np.int64),
            "rnx_vec": receiver_points[:, 0] + 1,
            "rnz_vec": receiver_points[:, 1] + 1,
            "rec_n": receiver_points.shape[0],
            "geometry_rec_type": "configured_2d_hess",
        },
    )


def save_model_plot(material: dict, normalized: dict, output_dir: Path) -> Path:
    """Save a nine-panel view of density, velocities, anisotropy, and stiffness."""
    plot, grid = normalized["plot"], normalized["grid"]
    source, receivers = normalized["source"], normalized["receivers"]
    diagnostics = normalized["hess"]["diagnostics"]
    extent = (0.0, (grid["nx"] - 1) * grid["dx"], (grid["nz"] - 1) * grid["dz"], 0.0)
    fields = (
        (material["rho"], "Density (kg/m3)", 1.0, plot["cmap"]),
        (diagnostics["vp_m_s"], "Vertical Vp (m/s)", 1.0, plot["cmap"]),
        (diagnostics["vs_m_s"], "Vertical Vs (m/s)", 1.0, plot["cmap"]),
        (diagnostics["epsilon"], "Thomsen epsilon", 1.0, plot["anisotropy_cmap"]),
        (diagnostics["delta"], "Thomsen delta", 1.0, plot["anisotropy_cmap"]),
        (material["C11"], "C11 (GPa)", 1.0e-9, plot["cmap"]),
        (material["C13"], "C13 (GPa)", 1.0e-9, plot["cmap"]),
        (material["C33"], "C33 (GPa)", 1.0e-9, plot["cmap"]),
        (material["C44"], "C44 (GPa)", 1.0e-9, plot["cmap"]),
    )
    figure, axes = plt.subplots(
        3, 3, figsize=(plot["fig_width"], plot["fig_height"]),
        constrained_layout=True, sharex=True, sharey=True,
    )
    source_x, source_z = source["x"] * grid["dx"], source["z"] * grid["dz"]
    receiver_x, receiver_z = receivers[:, 0] * grid["dx"], receivers[:, 1] * grid["dz"]
    for axis, (array, title, scale, cmap) in zip(axes.reshape(-1), fields):
        image = axis.imshow(array * scale, origin="upper", extent=extent, aspect="auto", cmap=cmap)
        axis.scatter(receiver_x, receiver_z, marker="v", s=7, c="white", edgecolors="#202124", linewidths=0.25)
        axis.scatter(source_x, source_z, marker="*", s=70, c="#ffd166", edgecolors="#202124", linewidths=0.6)
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        figure.colorbar(image, ax=axis, shrink=0.78, pad=0.02)
    for axis in axes[:, 0]:
        axis.set_ylabel("Z (m, depth)")
    figure.suptitle("Hess 2-D VTI model", fontsize=14)
    path = output_dir / plot["filename"]
    figure.savefig(path, dpi=plot["dpi"], bbox_inches="tight")
    plt.close(figure)
    return path


def _run_script(project_root: Path, normalized: dict, backend: str) -> str:
    """Create a PowerShell launcher using the normalized forward options."""
    forward = normalized["forward"]
    return f'''$ErrorActionPreference = "Stop"
$Python = "{forward['python_executable']}"
$CandidateProjectRoot = Join-Path $PSScriptRoot "..\.."
if (Test-Path -LiteralPath (Join-Path $CandidateProjectRoot "main.py")) {{
  $ProjectRoot = (Resolve-Path -LiteralPath $CandidateProjectRoot).Path
}} else {{
  $ProjectRoot = "{project_root}"
}}
# main.py places runtime caches in the shared upload-directory .cache folder.
& $Python (Join-Path $ProjectRoot "main.py") `
  --case-dir $PSScriptRoot `
  --backend {backend} `
  --source case `
  --order {forward['order']} `
  --dtype {forward['dtype']} `
  --block {forward['block']} `
  --record-stride {forward['record_stride']} `
  --snapshot-step {forward['snapshot_step']} `
  --output-dir (Join-Path $PSScriptRoot "{forward['output_dir']}") `
  --output-prefix "{forward['output_prefix']}"
'''


def build_case(config_path: str | Path, output_override: str | None = None, overwrite: bool = False) -> dict:
    """Export a validated case and refuse existing outputs unless overwrite is set."""
    config, resolved_config_path = load_config(config_path)
    normalized, material = validate_config(config, resolved_config_path)
    output_dir = resolve_output_dir(config, resolved_config_path, output_override)
    generated_names = set(GENERATED_FILES)
    generated_names.add(normalized["plot"]["filename"])
    existing = [output_dir / name for name in generated_names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output files already exist; use --overwrite: {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pml = build_pml(normalized, material["rho"].dtype)
    source_values = ricker(**{
        name: normalized["source"][name]
        for name in ("total_time", "dt", "f0", "t0", "scale")
    })
    source_payload = {
        "src_name": "Ricker", "source_kind": normalized["source"]["kind"],
        "T": normalized["source"]["total_time"], "dt": normalized["source"]["dt"],
        "f0": normalized["source"]["f0"], "t0": normalized["source"]["t0"],
        "src_scale": normalized["source"]["scale"], "src": source_values["src"],
        "t": source_values["t"], "N": source_values["N"], "Ns": source_values["N"],
    }
    grid = normalized["grid"]
    model_payload = {
        "nx": grid["nx"], "nz": grid["nz"], "dx": grid["dx"], "dz": grid["dz"],
        "rhom": material["rho"],
        "vpm": normalized["hess"]["diagnostics"]["vp_m_s"],
        "vsm": normalized["hess"]["diagnostics"]["vs_m_s"],
        "epsilon": normalized["hess"]["diagnostics"]["epsilon"],
        "delta": normalized["hess"]["diagnostics"]["delta"],
        "wave_type": "Elastic-Acoustic-2D-qP-qSV",
        "m_name": "Hess_VTI_2D",
        **{name: material[name] for name in INDEPENDENT_VTI_FIELDS},
    }
    source_geometry, receiver_geometry = geometry_payloads(normalized)
    compression = normalized["mat_compression"]
    savemat(output_dir / "model.mat", model_payload, do_compression=compression)
    savemat(output_dir / "source.mat", source_payload, do_compression=compression)
    savemat(
        output_dir / "BC.mat",
        {
            "BC": pml,
            "nAB": normalized["pml"]["nAB"],
            "BCtype": "topABC",
            "BCname": "PML",
            "reflection_coefficient": normalized["pml"]["reflection_coefficient"],
            "polynomial_power": normalized["pml"]["polynomial_power"],
        },
        do_compression=compression,
    )
    savemat(output_dir / "geometry_src.mat", source_geometry, do_compression=compression)
    savemat(output_dir / "geometry_rec.mat", receiver_geometry, do_compression=compression)

    config_used = {**config, "output_case_dir": str(output_dir)}
    (output_dir / "model_config_used.json").write_text(
        json.dumps(config_used, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    hess = normalized["hess"]
    summary = {
        "output_case_dir": str(output_dir),
        "hess_input_directory": str(hess["directory"]),
        "hess_parameterization": hess["parameterization"],
        "hess_transform": hess["transform"],
        "hess_scales": hess["scales"],
        "C66_source": hess["c66_source"],
        "shape_zx": [grid["nz"], grid["nx"]],
        "spacing_xz_m": [grid["dx"], grid["dz"]],
        "source": normalized["source"],
        "receiver_count": int(normalized["receivers"].shape[0]),
        "pml": normalized["pml"],
        "validation": normalized["validation"],
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    project_root = Path(__file__).resolve().parent
    (output_dir / "run_gpu.ps1").write_text(
        _run_script(project_root, normalized, "gpu"), encoding="utf-8-sig"
    )
    (output_dir / "run_cpu.ps1").write_text(
        _run_script(project_root, normalized, "cpu"), encoding="utf-8-sig"
    )
    plot_path = save_model_plot(material, normalized, output_dir) if normalized["plot"]["enabled"] else None
    return {"output_dir": output_dir, "plot": plot_path, "summary": summary}


def main() -> None:
    """Parse builder options, generate the case, and report its location."""
    parser = argparse.ArgumentParser(description="Build a 2-D Hess VTI forward-modelling case")
    parser.add_argument("--config", default="hess_vti2d_config.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_case(args.config, args.output_dir, args.overwrite)
    print("case saved:", result["output_dir"])
    if result["plot"] is not None:
        print("model figure:", result["plot"])
    print("shape [z,x]:", result["summary"]["shape_zx"])
    print("CFL dt limit:", result["summary"]["validation"]["cfl_dt_limit_s"])


if __name__ == "__main__":
    main()

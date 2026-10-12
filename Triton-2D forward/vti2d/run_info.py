"""Format console summaries from the actual 2-D case and runtime options.

Reports include material ranges, geometry, time sampling, and run timing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _number(value: float) -> str:
    return f"{float(value):.9g}"


def _optional(value: float | None, unit: str = "") -> str:
    if value is None:
        return "not stored in case (rebuild the case to add it)"
    suffix = f" {unit}" if unit else ""
    return f"{_number(value)}{suffix}"


def _line(label: str, value) -> str:
    return f"{label:<31}: {value}"


def _range_text(values: np.ndarray, scale: float = 1.0, unit: str = "") -> str:
    array = np.asarray(values, dtype=np.float64)
    suffix = f" {unit}" if unit else ""
    return f"[{_number(np.min(array) * scale)}, {_number(np.max(array) * scale)}]{suffix}"


def format_run_configuration(case, args, source_kind: str, region_stats: dict) -> str:
    """Return a self-contained summary of values actually used by this run."""
    if args.record_stride < 1:
        raise ValueError("record_stride must be positive")
    snapshot_step = case.nt - 3 if args.snapshot_step == -1 else args.snapshot_step
    snapshot_time_s = (snapshot_step - 1) * case.dt
    pml_mask = (np.asarray(case.pmlx) != 0.0) | (np.asarray(case.pmlz) != 0.0)
    rec_x = np.asarray(case.rec_x, dtype=np.int64)
    rec_z = np.asarray(case.rec_z, dtype=np.int64)
    receiver_layout = "horizontal line" if np.all(rec_z == rec_z[0]) else "explicit/non-horizontal"
    cfl_limit = float(case.validation["cfl_dt_limit_s"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    plot_status = "disabled (--no-plots)" if args.no_plots else (
        f"enabled, cmap={args.plot_cmap}, dpi={args.plot_dpi}"
    )

    lines = [
        "=" * 76,
        "VERSION 12: 2-D VTI FORWARD RUN CONFIGURATION (actual runtime values)",
        "=" * 76,
        "[Case and model]",
        _line("Case directory", case.case_dir),
        _line("Material model", "Hess-VTI-2D qP-qSV"),
        _line("Grid shape [nz,nx]", f"[{case.nz}, {case.nx}]"),
        _line("Grid spacing [dx,dz]", f"[{_number(case.dx)}, {_number(case.dz)}] m"),
        _line(
            "Grid physical span [X,Z]",
            f"[{_number((case.nx - 1) * case.dx)}, {_number((case.nz - 1) * case.dz)}] m",
        ),
        _line("Density rho range", _range_text(case.rho, unit="kg/m^3")),
        _line("C11 range", _range_text(case.c11, 1.0e-9, "GPa")),
        _line("C13 range", _range_text(case.c13, 1.0e-9, "GPa")),
        _line("C33 range", _range_text(case.c33, 1.0e-9, "GPa")),
        _line("C44 range", _range_text(case.c44, 1.0e-9, "GPa")),
        _line("C66 range", _range_text(case.c66, 1.0e-9, "GPa")),
        "[Time sampling]",
        _line("Simulation dt", f"{_number(case.dt * 1.0e3)} ms ({_number(case.dt)} s)"),
        _line(
            "Receiver sample interval",
            f"{_number(case.dt * args.record_stride * 1.0e3)} ms "
            f"(record_stride={args.record_stride})",
        ),
        _line("Configured total time", f"{_number(case.total_time * 1.0e3)} ms"),
        _line("Source sample count nt", case.nt),
        _line("Nominal sampled duration", f"{_number((case.nt - 1) * case.dt * 1.0e3)} ms"),
        _line("Solver update count", case.nt - 3),
        _line("Requested snapshot step", args.snapshot_step),
        _line("Resolved snapshot step", snapshot_step),
        _line("Resolved snapshot time", f"{_number(snapshot_time_s * 1.0e3)} ms"),
        "[Source]",
        _line("Wavelet", case.source_name),
        _line("Active source mechanism", source_kind),
        _line("Mechanism stored in case", case.default_source_kind),
        _line("Main frequency f0", _optional(case.source_f0_hz, "Hz")),
        _line("Wavelet delay t0", _optional(
            None if case.source_t0_s is None else case.source_t0_s * 1.0e3, "ms"
        )),
        _line("Source scale", _optional(case.source_scale)),
        _line("Source index [x,z]", f"[{case.src_x}, {case.src_z}] (zero-based)"),
        _line(
            "Source position [X,Z]",
            f"[{_number(case.src_x * case.dx)}, {_number(case.src_z * case.dz)}] m",
        ),
        "[PML boundary]",
        _line("Boundary name/type", f"{case.pml_name} / {case.bc_type}"),
        _line("PML thickness nAB", f"{case.nab} grid points on each side"),
        _line(
            "PML physical thickness [X,Z]",
            f"[{_number(case.nab * case.dx)}, {_number(case.nab * case.dz)}] m",
        ),
        _line("Target reflection coefficient", _optional(case.pml_reflection_coefficient)),
        _line("Damping polynomial power", _optional(case.pml_polynomial_power)),
        _line("PML cells (full grid)", int(np.count_nonzero(pml_mask))),
        _line("PML damping max [x,z]", f"[{_number(np.max(case.pmlx))}, {_number(np.max(case.pmlz))}] 1/s"),
        _line("FD-active points", region_stats.get("fd_active_points", "unavailable")),
        _line("FD-active interior points", region_stats.get("interior_points", "unavailable")),
        _line("FD-active PML points", region_stats.get("pml_points", "unavailable")),
        "[Receivers]",
        _line("Receiver count", case.nrec),
        _line("Receiver layout", receiver_layout),
        _line(
            "Receiver index range [x,z]",
            f"x=[{int(np.min(rec_x))},{int(np.max(rec_x))}], "
            f"z=[{int(np.min(rec_z))},{int(np.max(rec_z))}]",
        ),
        _line(
            "Receiver position range [X,Z]",
            f"X=[{_number(np.min(rec_x) * case.dx)},{_number(np.max(rec_x) * case.dx)}] m, "
            f"Z=[{_number(np.min(rec_z) * case.dz)},{_number(np.max(rec_z) * case.dz)}] m",
        ),
        "[Numerics and output]",
        _line("Backend / dtype", f"{args.backend} / {args.dtype}"),
        _line("FD spatial order", args.order),
        _line("Device", args.device if args.backend == "gpu" else "CPU"),
        _line("Triton block", args.block if args.backend == "gpu" else "not used by CPU"),
        _line("CUDA sync debug", bool(args.sync_debug) if args.backend == "gpu" else "not used by CPU"),
        _line("Conservative vmax", f"{_number(case.validation['conservative_vmax_m_s'])} m/s"),
        _line("CFL dt limit", f"{_number(cfl_limit * 1.0e3)} ms"),
        _line("dt / CFL limit", _number(case.dt / cfl_limit)),
        _line("Output directory", output_dir),
        _line("Output prefix", args.output_prefix),
        _line("Plotting", plot_status),
        "=" * 76,
    ]
    return "\n".join(str(line) for line in lines)


def format_run_completion(result, case) -> str:
    """Summarize runtime, saved samples, snapshot timing, and near-source geometry."""
    diagnostics = result["diagnostics"]
    record_stride = int(diagnostics.get("record_stride", 1))
    sample_count = int(np.asarray(result["Sx"]).shape[0])
    lines = [
        "=" * 76,
        "VERSION 12: 2-D VTI FORWARD RUN COMPLETED",
        "=" * 76,
        _line("Runtime", f"{_number(result['runtime_s'])} s"),
        _line("Saved receiver samples", sample_count),
        _line("Receiver sample interval", f"{_number(case.dt * record_stride * 1.0e3)} ms"),
        _line("Snapshot step", int(diagnostics["snapshot_step"])),
        _line("Snapshot time", f"{_number(float(diagnostics['snapshot_time']) * 1.0e3)} ms"),
        _line(
            "Near-source receiver index",
            f"[{int(diagnostics['near_rec_x'])}, {int(diagnostics['near_rec_z'])}]",
        ),
        _line(
            "Near-source receiver position",
            f"[{_number(int(diagnostics['near_rec_x']) * case.dx)}, "
            f"{_number(int(diagnostics['near_rec_z']) * case.dz)}] m",
        ),
        "=" * 76,
    ]
    return "\n".join(str(line) for line in lines)

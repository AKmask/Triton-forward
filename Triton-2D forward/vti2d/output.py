"""Export 2-D receiver data and wavefield diagnostics to MAT, NPZ, and JSON.

Stored dt includes record_stride; simulation_dt retains the solver step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import savemat


def save_result(result, case, output_dir, stem, order, backend, dtype, source_kind, record_stride):
    """Save receiver records, material fields, diagnostics, and runtime metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "Sx": result["Sx"],
        "Sz": result["Sz"],
        "dt": case.dt * record_stride,
        "simulation_dt": case.dt,
        "N": case.nt,
        "M": order,
        "dN_SS": record_stride,
        "dx": case.dx,
        "dz": case.dz,
        "nx": case.nx,
        "nz": case.nz,
        "src_x": case.src_x,
        "src_z": case.src_z,
        "rec_x": case.rec_x,
        "rec_z": case.rec_z,
        "rec_x_m": case.rec_x * case.dx,
        "rec_z_m": case.rec_z * case.dz,
        "rho": case.rho,
        "C11": case.c11,
        "C13": case.c13,
        "C33": case.c33,
        "C44": case.c44,
        "C66": case.c66,
        "material_model": "Hess-VTI-2D-qP-qSV",
        "source_kind": source_kind,
        "backend": backend,
    }
    payload.update(result["diagnostics"])
    mat_path = output / f"{stem}.mat"
    npz_path = output / f"{stem}.npz"
    savemat(mat_path, payload, do_compression=False)
    np.savez(npz_path, **payload)
    metadata = {
        "backend": backend,
        "dtype": dtype,
        "runtime_s": float(result["runtime_s"]),
        "shape_zx": [case.nz, case.nx],
        "spacing_xz_m": [case.dx, case.dz],
        "nt": case.nt,
        "simulation_dt_s": case.dt,
        "record_stride": record_stride,
        "order": order,
        "source_kind": source_kind,
        "source_xz_index": [case.src_x, case.src_z],
        "near_receiver_xz_index": [
            int(result["diagnostics"]["near_rec_x"]),
            int(result["diagnostics"]["near_rec_z"]),
        ],
        "snapshot_step": int(result["diagnostics"]["snapshot_step"]),
        "snapshot_time_s": float(result["diagnostics"]["snapshot_time"]),
        "kernel_design": result.get("kernel_design"),
        "state_memory_bytes": result.get("state_memory_bytes"),
        "region_stats": result.get("region_stats", {}),
        "vti_validation": case.validation,
        "model_family": "Hess VTI",
    }
    metadata_path = output / f"{stem}_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"mat": mat_path, "npz": npz_path, "meta": metadata_path}

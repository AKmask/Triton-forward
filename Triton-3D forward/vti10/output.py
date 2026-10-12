"""Write 3-D receiver records and snapshot diagnostics to MAT, NPZ, and JSON.

The record stride is fixed at one to preserve FDwave3D comparison layout.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.io import savemat


def save_result(result, case, outdir, stem, order, mode, dtype):
    """Write MAT/NPZ payloads and a JSON summary; return the MAT file path."""
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    payload={"Sx":result["Sx"],"Sy":result["Sy"],"Sz":result["Sz"],"dt":case.dt,"N":case.nt,"M":order,
             "dx":case.dx,"dy":case.dy,"dz":case.dz,"nx":case.nx,"ny":case.ny,"nz":case.nz,"dN_SS":1,
             "material_model":case.material_model,
             "vti_independent_coefficients":np.asarray(["C11","C13","C33","C44","C66"],dtype=object)}
    payload.update(result["diagnostics"])
    savemat(out/f"{stem}.mat",payload,do_compression=False)
    np.savez(out/f"{stem}.npz",**payload)
    meta={"mode":mode,"dtype":dtype,"runtime_s":float(result["runtime_s"]),"N":case.nt,"M":order,
          "shape":[case.nz,case.ny,case.nx],"nrec":case.nrec,"nab":case.nab,"bc_type":case.bc_type,"bc_name":case.bc_name,
          "snapshot_step":int(result["diagnostics"]["snapshot_step"]),
          "snapshot_time":float(result["diagnostics"]["snapshot_time"]),
          "source_xyz":[case.src_x,case.src_y,case.src_z],
          "near_receiver_xyz":[int(result["diagnostics"]["near_rec_x"]),case.src_y,case.src_z],
          "kernel_design":result.get("kernel_design"),"region_stats":result.get("region_stats",{}),
          "material_model":case.material_model,"stiffness_source":case.stiffness_source,
          "vti_independent_coefficients":["C11","C13","C33","C44","C66"],
          "vti_validation":case.vti_validation}
    (out/f"{stem}_meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    return out/f"{stem}.mat"

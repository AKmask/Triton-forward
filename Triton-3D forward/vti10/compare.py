"""Compare FDwave3D and Triton Sx/Sy/Sz receiver records from MAT files.

Reports include relative L2 error, correlation, and amplitude-scaled error.
MATLAB v7.3 reference results are read through the optional h5py dependency.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def _read_v73(path):
    """Read top-level HDF5 datasets and restore MATLAB dimension order."""
    import h5py
    out={}
    with h5py.File(path,"r") as f:
        for k in f.keys():
            obj=f[k]
            if isinstance(obj,h5py.Dataset):
                a=np.array(obj)
                # Reverse HDF5 dimension order to restore MATLAB array orientation.
                if a.ndim>=2:
                    a=np.transpose(a,tuple(reversed(range(a.ndim))))
                out[k]=np.asarray(a).squeeze()
    return out


def _read(path):
    """Try SciPy MAT loading before falling back to the MATLAB v7.3 reader."""
    path=Path(path)
    try:
        return loadmat(path,squeeze_me=True)
    except (NotImplementedError,ValueError,OSError):
        return _read_v73(path)


def compare(fd,triton,allow_truncate=False):
    """Compare Sx/Sy/Sz arrays; truncate mismatched shapes only when requested."""
    a=_read(fd); b=_read(triton); out={}
    for k in ("Sx","Sy","Sz"):
        if k not in a or k not in b:
            raise KeyError(f"A result file is missing variable {k}")
        x=np.asarray(a[k],dtype=np.float64); y=np.asarray(b[k],dtype=np.float64)
        if x.ndim==1: x=x[:,None]
        if y.ndim==1: y=y[:,None]
        if x.shape!=y.shape:
            if not allow_truncate:
                raise ValueError(f"{k} shape mismatch: FD={x.shape}, Triton={y.shape}. Match N, DN_SS, and receiver counts first.")
            n0=min(x.shape[0],y.shape[0]); n1=min(x.shape[1],y.shape[1]); x=x[:n0,:n1]; y=y[:n0,:n1]
        den=np.linalg.norm(x.ravel())+1e-30
        rel=float(np.linalg.norm((y-x).ravel())/den)
        corr=float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if np.std(x)>0 and np.std(y)>0 else float("nan")
        alpha=float(np.dot(x.ravel(),y.ravel())/(np.dot(y.ravel(),y.ravel())+1e-30))
        scaled=float(np.linalg.norm((alpha*y-x).ravel())/den)
        out[k]={"relative_l2":rel,"max_abs":float(np.max(np.abs(y-x))),"corrcoef":corr,
                "best_amplitude_scale":alpha,"scaled_relative_l2":scaled,"shape_used":list(x.shape)}
    return out

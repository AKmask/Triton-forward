"""Coordinate region-fused 3-D VTI velocity-stress propagation on CUDA.

The solver advances nt-3 steps and records three velocity components,
a near-source trace, and XY/XZ/YZ snapshots through the source.
"""

from __future__ import annotations
import time
import numpy as np
import torch
from .fd_coeff import padded_coefficients
from .region import build_region_indices, region_stats
from .source import stress_source_weights
from .kernels.region_common import launch_interior_velocity, launch_interior_stress
from .kernels.sparse_ops import launch_source, launch_sample
from .kernels.split_pml import launch_pml_velocity, launch_pml_stress, launch_split_source


class RegionFusedSplitPMLSolver:
    """Optimizations 3+4: separate interior/PML data flow with fused derivatives, recurrences, and reconstruction."""
    def __init__(self, case, order=10, dtype="fp64", device="cuda", block=256, sync_debug=False):
        if not torch.cuda.is_available(): raise RuntimeError("A CUDA GPU is required")
        self.case=case; self.order=int(order); self.half=self.order//2; self.device=torch.device(device); self.block=int(block)
        self.sync_debug=bool(sync_debug)
        self.tdtype=torch.float64 if dtype=="fp64" else torch.float32
        self.npdtype=np.float64 if dtype=="fp64" else np.float32
        self.coeffs=padded_coefficients(order); self.shape=case.shape; self.spacing=(case.dx,case.dy,case.dz)
        self.fields=[torch.zeros(self.shape,dtype=self.tdtype,device=self.device) for _ in range(9)]
        self.vx,self.vy,self.vz,self.sxx,self.syy,self.szz,self.sxy,self.sxz,self.syz=self.fields
        self.bx=self._gpu(case.bx); self.by=self._gpu(case.by); self.bz=self._gpu(case.bz)
        self.cij=[self._gpu(a) for a in (case.c11,case.c12,case.c13,case.c22,case.c23,case.c33,case.c44s,case.c55s,case.c66s)]
        self.pml=[self._gpu(case.pmlx),self._gpu(case.pmly),self._gpu(case.pmlz)]
        interior,pml=build_region_indices(case,self.half)
        self.interior_idx=torch.from_numpy(interior).to(self.device); self.pml_idx=torch.from_numpy(pml).to(self.device)
        self.stats=region_stats(case,self.half)
        # Retain all 24 split states without compression, as in version 08; the 08-to-09 comparison isolates optimization 4.
        self.split_v=[torch.zeros(self.shape,dtype=self.tdtype,device=self.device) for _ in range(9)]
        self.split_s=[torch.zeros(self.shape,dtype=self.tdtype,device=self.device) for _ in range(15)]
        # Flatten C-contiguous [z, y, x] coordinates; x is the fastest axis.
        rec=(case.rec_z*case.ny*case.nx + case.rec_y*case.nx + case.rec_x).astype(np.int64)
        self.rec_idx=torch.from_numpy(rec).to(self.device)
        self.src_idx=int(case.src_z*case.ny*case.nx + case.src_y*case.nx + case.src_x)
        self.near_rec_x=case.src_x + 1 if case.src_x + 1 < case.nx else case.src_x - 1
        if self.near_rec_x < 0:
            raise ValueError("A near-source receiver cannot be placed when nx is less than 2")
        near_rec=(case.src_z*case.ny*case.nx + case.src_y*case.nx + self.near_rec_x)
        self.near_rec_idx=torch.tensor([near_rec],dtype=torch.int64,device=self.device)
        self.source=torch.from_numpy(np.asarray(case.source,dtype=self.npdtype)).to(self.device)
        self.source_is_pml=bool((case.pmlx[case.src_z,case.src_y,case.src_x]!=0.0) or
                                (case.pmly[case.src_z,case.src_y,case.src_x]!=0.0) or
                                (case.pmlz[case.src_z,case.src_y,case.src_x]!=0.0))

    def _gpu(self,a): return torch.from_numpy(np.ascontiguousarray(a,dtype=self.npdtype)).to(self.device)
    def _sync(self):
        if self.sync_debug: torch.cuda.synchronize()

    def run(self,kind="dc1",record_stride=1,snapshot_step=-1):
        """Advance one source run and return receiver records and source-plane snapshots."""
        if record_stride!=1: raise NotImplementedError("Strict FDwave3D comparison supports record_stride=1 only")
        n=self.case.nt; nr=self.case.nrec
        if snapshot_step == -1: snapshot_step=n-3
        if not 1 <= snapshot_step <= n-3:
            raise ValueError(f"snapshot_step must be in [1, {n-3}], got {snapshot_step}")
        ox=torch.zeros((n,nr),dtype=self.tdtype,device=self.device); oy=torch.zeros_like(ox); oz=torch.zeros_like(ox)
        near_ox=torch.zeros((n,1),dtype=self.tdtype,device=self.device)
        near_oy=torch.zeros_like(near_ox); near_oz=torch.zeros_like(near_ox)
        weights=stress_source_weights(kind,self.npdtype)
        plane_snapshots=None
        torch.cuda.synchronize(); t0=time.perf_counter()
        for it in range(n-3):
            launch_source(tuple(self.fields[3:]),self.source,it,self.src_idx,weights); self._sync()
            # PML reconstruction must retain the stress injected at the source.
            if self.source_is_pml:
                launch_split_source(self.split_s,self.source,it,self.src_idx,weights); self._sync()

            # Update interior velocities directly; fuse directional derivatives, split updates, and velocity reconstruction in PML.
            launch_interior_velocity(self.interior_idx,self.fields,(self.bx,self.by,self.bz),self.shape,self.spacing,
                                     self.case.dt,self.coeffs,self.half,self.block); self._sync()
            launch_pml_velocity(self.pml_idx,self.fields,(self.bx,self.by,self.bz),self.split_v,self.pml,self.shape,self.spacing,
                                self.case.dt,self.coeffs,self.half,self.block); self._sync()

            # Update interior stresses directly; fuse directional derivatives, split updates, and stress reconstruction in PML.
            launch_interior_stress(self.interior_idx,self.fields,self.cij,self.shape,self.spacing,self.case.dt,
                                   self.coeffs,self.half,self.block); self._sync()
            launch_pml_stress(self.pml_idx,self.fields,self.cij,self.split_s,self.pml,self.shape,self.spacing,
                              self.case.dt,self.coeffs,self.half,self.block); self._sync()

            launch_sample((self.vx,self.vy,self.vz),self.rec_idx,(ox,oy,oz),it+1); self._sync()
            launch_sample((self.vx,self.vy,self.vz),self.near_rec_idx,(near_ox,near_oy,near_oz),it+1); self._sync()
            if it+1 == snapshot_step:
                plane_snapshots={}
                for component,field in (("vx",self.vx),("vy",self.vy),("vz",self.vz)):
                    # Shared solver plane orientation: XY=(ny,nx), XZ=(nz,nx), YZ=(nz,ny).
                    plane_snapshots[f"snapshot_{component}_xy"]=field[self.case.src_z,:,:].clone()
                    plane_snapshots[f"snapshot_{component}_xz"]=field[:,self.case.src_y,:].clone()
                    plane_snapshots[f"snapshot_{component}_yz"]=field[:,:,self.case.src_x].clone()
        torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
        diagnostics={key:value.cpu().numpy() for key,value in plane_snapshots.items()}
        diagnostics.update(
            near_Sx=near_ox.cpu().numpy(),near_Sy=near_oy.cpu().numpy(),near_Sz=near_oz.cpu().numpy(),
            snapshot_step=snapshot_step,snapshot_time=(snapshot_step-1)*self.case.dt,
            src_x=self.case.src_x,src_y=self.case.src_y,src_z=self.case.src_z,
            near_rec_x=self.near_rec_x,near_rec_y=self.case.src_y,near_rec_z=self.case.src_z,
        )
        return dict(Sx=ox.cpu().numpy(),Sy=oy.cpu().numpy(),Sz=oz.cpu().numpy(),runtime_s=elapsed,
                    diagnostics=diagnostics,
                    region_stats=self.stats,kernel_design="region-fused-splitpml-vti")

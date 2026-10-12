"""Coordinate CUDA/Triton propagation for a 2-D VTI case.

Fields use contiguous [z, x] arrays. Interior and PML kernels operate on
disjoint index lists; receivers store sampled particle velocities.
"""

from __future__ import annotations

import time
import numpy as np
import torch

from .fd_coeff import padded_coefficients
from .kernels.accelerated import (
    launch_interior_stress,
    launch_interior_velocity,
    launch_pml_stress,
    launch_pml_velocity,
)
from .kernels.sparse import launch_sample, launch_source, launch_split_source
from .region import build_region_indices, region_stats
from .source import stress_source_weights


class TritonRegionFusedSolver:
    """CUDA/Triton 2-D VTI solver with heterogeneous interior/PML data flow."""

    def __init__(self, case, order=10, dtype="fp64", device="cuda", block=256, sync_debug=False):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the Triton backend")
        if dtype not in {"fp32", "fp64"}:
            raise ValueError("dtype must be fp32 or fp64")
        if block not in {64, 128, 256, 512, 1024}:
            raise ValueError("block must be one of 64, 128, 256, 512, 1024")
        self.case = case
        self.order = int(order)
        self.half = self.order // 2
        self.device = torch.device(device)
        self.block = int(block)
        self.sync_debug = bool(sync_debug)
        self.torch_dtype = torch.float64 if dtype == "fp64" else torch.float32
        self.numpy_dtype = np.float64 if dtype == "fp64" else np.float32
        self.coefficients = padded_coefficients(self.order)
        self.fields = [torch.zeros(case.shape, dtype=self.torch_dtype, device=self.device) for _ in range(5)]
        self.vx, self.vz, self.sxx, self.szz, self.sxz = self.fields
        self.inverse_density = (self._gpu(case.bx), self._gpu(case.bz))
        self.stiffness = tuple(self._gpu(array) for array in (case.c11, case.c13, case.c33, case.c55s))
        self.pml = (self._gpu(case.pmlx), self._gpu(case.pmlz))
        interior, pml = build_region_indices(case, self.half)
        self.interior_indices = torch.from_numpy(interior).to(self.device)
        self.pml_indices = torch.from_numpy(pml).to(self.device)
        self.stats = region_stats(case, self.half)
        # Two directional splits per component: 4 velocity and 6 stress states.
        self.split_velocity = [torch.zeros_like(self.vx) for _ in range(4)]
        self.split_stress = [torch.zeros_like(self.vx) for _ in range(6)]
        # C-contiguous [z, x] storage maps each point to z*nx+x.
        receiver_indices = (case.rec_z * case.nx + case.rec_x).astype(np.int64)
        self.receiver_indices = torch.from_numpy(receiver_indices).to(self.device)
        self.source_index = int(case.src_z * case.nx + case.src_x)
        self.near_x = case.src_x + 1 if case.src_x + 1 < case.nx - self.half else case.src_x - 1
        if self.near_x < 0:
            raise ValueError("nx must be at least 2 for a near-source receiver")
        near_index = np.asarray([case.src_z * case.nx + self.near_x], dtype=np.int64)
        self.near_receiver_index = torch.from_numpy(near_index).to(self.device)
        self.source = torch.from_numpy(np.asarray(case.source, dtype=self.numpy_dtype)).to(self.device)
        self.source_is_pml = bool(
            case.pmlx[case.src_z, case.src_x] != 0.0 or case.pmlz[case.src_z, case.src_x] != 0.0
        )
        element_size = torch.empty((), dtype=self.torch_dtype).element_size()
        self.state_memory_bytes = int(case.nx * case.nz * (5 + 4 + 6) * element_size)

    def _gpu(self, array):
        contiguous = np.ascontiguousarray(array, dtype=self.numpy_dtype)
        return torch.from_numpy(contiguous).to(self.device)

    def _sync(self):
        if self.sync_debug:
            torch.cuda.synchronize(self.device)

    def run(self, source_kind="explosive", record_stride=1, snapshot_step=-1):
        """Advance the allocated GPU fields and transfer records and diagnostics to NumPy."""
        case = self.case
        if record_stride < 1:
            raise ValueError("record_stride must be positive")
        if snapshot_step == -1:
            snapshot_step = case.nt - 3
        if not 1 <= snapshot_step <= case.nt - 3:
            raise ValueError(f"snapshot_step must be in [1,{case.nt - 3}]")
        record_count = (case.nt - 1) // record_stride + 1
        sx = torch.zeros((record_count, case.nrec), dtype=self.torch_dtype, device=self.device)
        sz = torch.zeros_like(sx)
        near_sx = torch.zeros((record_count, 1), dtype=self.torch_dtype, device=self.device)
        near_sz = torch.zeros_like(near_sx)
        weights = stress_source_weights(source_kind, self.numpy_dtype)
        snapshot = None
        spacing = (case.dx, case.dz)

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        for iteration in range(case.nt - 3):
            launch_source((self.sxx, self.szz, self.sxz), self.source, iteration, self.source_index, weights)
            self._sync()
            # Retain the source contribution when split stresses are reconstructed.
            if self.source_is_pml:
                launch_split_source(self.split_stress, self.source, iteration, self.source_index, weights)
                self._sync()
            # Advance velocities first, then stresses, on the two disjoint regions.
            launch_interior_velocity(
                self.interior_indices, self.fields, self.inverse_density, case.shape, spacing,
                case.dt, self.coefficients, self.half, self.block,
            )
            self._sync()
            launch_pml_velocity(
                self.pml_indices, self.fields, self.inverse_density, self.split_velocity, self.pml,
                case.shape, spacing, case.dt, self.coefficients, self.half, self.block,
            )
            self._sync()
            launch_interior_stress(
                self.interior_indices, self.fields, self.stiffness, case.shape, spacing,
                case.dt, self.coefficients, self.half, self.block,
            )
            self._sync()
            launch_pml_stress(
                self.pml_indices, self.fields, self.stiffness, self.split_stress, self.pml,
                case.shape, spacing, case.dt, self.coefficients, self.half, self.block,
            )
            self._sync()
            step = iteration + 1
            if step % record_stride == 0:
                row = step // record_stride
                launch_sample((self.vx, self.vz), self.receiver_indices, (sx, sz), row, block=self.block)
                launch_sample((self.vx, self.vz), self.near_receiver_index, (near_sx, near_sz), row, block=64)
                self._sync()
            if step == snapshot_step:
                snapshot = {"snapshot_vx": self.vx.clone(), "snapshot_vz": self.vz.clone()}
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        if snapshot is None:
            raise RuntimeError("snapshot was not captured")
        diagnostics = {
            **{name: value.cpu().numpy() for name, value in snapshot.items()},
            "near_Sx": near_sx.cpu().numpy(),
            "near_Sz": near_sz.cpu().numpy(),
            "snapshot_step": snapshot_step,
            "snapshot_time": (snapshot_step - 1) * case.dt,
            "src_x": case.src_x,
            "src_z": case.src_z,
            "near_rec_x": self.near_x,
            "near_rec_z": case.src_z,
            "record_stride": record_stride,
        }
        return {
            "Sx": sx.cpu().numpy(),
            "Sz": sz.cpu().numpy(),
            "runtime_s": elapsed,
            "diagnostics": diagnostics,
            "region_stats": self.stats,
            "kernel_design": "triton-region-fused-splitpml-vti2d",
            "state_memory_bytes": self.state_memory_bytes,
        }

"""NumPy reference propagation for the 2-D VTI velocity-stress scheme.

Interior cells use direct updates; absorbing cells use separate x/z
PML states. Only cells with a complete finite-difference stencil evolve.
"""

from __future__ import annotations

import time
import numpy as np

from .fd_coeff import staggered_diff_coeff
from .region import region_masks, region_stats
from .source import stress_source_weights


def _fwd(field: np.ndarray, axis: int, spacing: float, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the forward staggered derivative along the selected array axis."""
    result = np.zeros_like(field)
    for index, coefficient in enumerate(coefficients):
        result += coefficient * (
            np.roll(field, -(index + 1), axis=axis) - np.roll(field, index, axis=axis)
        )
    return result / spacing


def _bwd(field: np.ndarray, axis: int, spacing: float, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the backward staggered derivative along the selected array axis."""
    result = np.zeros_like(field)
    for index, coefficient in enumerate(coefficients):
        result += coefficient * (
            np.roll(field, -index, axis=axis) - np.roll(field, index + 1, axis=axis)
        )
    return result / spacing


class CPUReferenceSolver:
    """Vectorized NumPy reference for the accelerated 2-D VTI scheme."""

    def __init__(self, case, order=10, dtype="fp64"):
        self.case = case
        self.order = int(order)
        self.half = self.order // 2
        self.dtype = np.float64 if dtype == "fp64" else np.float32
        self.coefficients = staggered_diff_coeff(self.order, dtype=self.dtype)
        self.interior_mask, self.pml_mask = region_masks(case, self.half)
        self.stats = region_stats(case, self.half)

    def run(self, source_kind="explosive", record_stride=1, snapshot_step=-1):
        """Advance nt-3 steps and return receiver records and one wavefield snapshot."""
        case = self.case
        if record_stride < 1:
            raise ValueError("record_stride must be positive")
        if snapshot_step == -1:
            snapshot_step = case.nt - 3
        if not 1 <= snapshot_step <= case.nt - 3:
            raise ValueError(f"snapshot_step must be in [1,{case.nt - 3}]")
        arrays = [np.zeros(case.shape, dtype=self.dtype) for _ in range(5)]
        vx, vz, sxx, szz, sxz = arrays
        split_velocity = [np.zeros_like(vx) for _ in range(4)]
        split_stress = [np.zeros_like(vx) for _ in range(6)]
        svx_x, svx_z, svz_x, svz_z = split_velocity
        ssxx_x, ssxx_z, sszz_x, sszz_z, ssxz_x, ssxz_z = split_stress
        bx = np.asarray(case.bx, dtype=self.dtype)
        bz = np.asarray(case.bz, dtype=self.dtype)
        c11 = np.asarray(case.c11, dtype=self.dtype)
        c13 = np.asarray(case.c13, dtype=self.dtype)
        c33 = np.asarray(case.c33, dtype=self.dtype)
        c55 = np.asarray(case.c55s, dtype=self.dtype)
        pmlx = np.asarray(case.pmlx, dtype=self.dtype)
        pmlz = np.asarray(case.pmlz, dtype=self.dtype)
        source = np.asarray(case.source, dtype=self.dtype)
        weights = stress_source_weights(source_kind, self.dtype)
        source_is_pml = bool(self.pml_mask[case.src_z, case.src_x])
        near_x = case.src_x + 1 if case.src_x + 1 < case.nx - self.half else case.src_x - 1
        record_count = (case.nt - 1) // record_stride + 1
        sx = np.zeros((record_count, case.nrec), dtype=self.dtype)
        sz = np.zeros_like(sx)
        near_sx = np.zeros((record_count, 1), dtype=self.dtype)
        near_sz = np.zeros_like(near_sx)
        snapshot = None
        interior, pml = self.interior_mask, self.pml_mask
        dt = self.dtype(case.dt)
        axn, axd = 1.0 - 0.5 * dt * pmlx, 1.0 + 0.5 * dt * pmlx
        azn, azd = 1.0 - 0.5 * dt * pmlz, 1.0 + 0.5 * dt * pmlz

        start = time.perf_counter()
        # Preserve the shared N-3 update convention and reserve output row zero.
        for iteration in range(case.nt - 3):
            amplitude = source[iteration]
            sxx[case.src_z, case.src_x] += amplitude * weights[0]
            szz[case.src_z, case.src_x] += amplitude * weights[1]
            sxz[case.src_z, case.src_x] += amplitude * weights[2]
            # Mirror injection in split states before PML reconstruction.
            if source_is_pml:
                ssxx_x[case.src_z, case.src_x] += 0.5 * amplitude * weights[0]
                ssxx_z[case.src_z, case.src_x] += 0.5 * amplitude * weights[0]
                sszz_x[case.src_z, case.src_x] += 0.5 * amplitude * weights[1]
                sszz_z[case.src_z, case.src_x] += 0.5 * amplitude * weights[1]
                ssxz_x[case.src_z, case.src_x] += 0.5 * amplitude * weights[2]
                ssxz_z[case.src_z, case.src_x] += 0.5 * amplitude * weights[2]

            # Velocity stage: interior updates and damped directional PML states.
            dx_sxx = _fwd(sxx, 1, case.dx, self.coefficients)
            dz_sxz = _bwd(sxz, 0, case.dz, self.coefficients)
            dx_sxz = _bwd(sxz, 1, case.dx, self.coefficients)
            dz_szz = _fwd(szz, 0, case.dz, self.coefficients)
            vx[interior] += dt * bx[interior] * (dx_sxx[interior] + dz_sxz[interior])
            vz[interior] += dt * bz[interior] * (dx_sxz[interior] + dz_szz[interior])
            svx_x[pml] = (axn[pml] * svx_x[pml] + dt * bx[pml] * dx_sxx[pml]) / axd[pml]
            svx_z[pml] = (azn[pml] * svx_z[pml] + dt * bx[pml] * dz_sxz[pml]) / azd[pml]
            svz_x[pml] = (axn[pml] * svz_x[pml] + dt * bz[pml] * dx_sxz[pml]) / axd[pml]
            svz_z[pml] = (azn[pml] * svz_z[pml] + dt * bz[pml] * dz_szz[pml]) / azd[pml]
            vx[pml] = svx_x[pml] + svx_z[pml]
            vz[pml] = svz_x[pml] + svz_z[pml]

            # Stress stage: apply VTI stiffnesses to the updated velocity derivatives.
            dx_vx = _bwd(vx, 1, case.dx, self.coefficients)
            dz_vz = _bwd(vz, 0, case.dz, self.coefficients)
            dx_vz = _fwd(vz, 1, case.dx, self.coefficients)
            dz_vx = _fwd(vx, 0, case.dz, self.coefficients)
            sxx[interior] += dt * (c11[interior] * dx_vx[interior] + c13[interior] * dz_vz[interior])
            szz[interior] += dt * (c13[interior] * dx_vx[interior] + c33[interior] * dz_vz[interior])
            sxz[interior] += dt * c55[interior] * (dx_vz[interior] + dz_vx[interior])
            ssxx_x[pml] = (axn[pml] * ssxx_x[pml] + dt * c11[pml] * dx_vx[pml]) / axd[pml]
            ssxx_z[pml] = (azn[pml] * ssxx_z[pml] + dt * c13[pml] * dz_vz[pml]) / azd[pml]
            sszz_x[pml] = (axn[pml] * sszz_x[pml] + dt * c13[pml] * dx_vx[pml]) / axd[pml]
            sszz_z[pml] = (azn[pml] * sszz_z[pml] + dt * c33[pml] * dz_vz[pml]) / azd[pml]
            ssxz_x[pml] = (axn[pml] * ssxz_x[pml] + dt * c55[pml] * dx_vz[pml]) / axd[pml]
            ssxz_z[pml] = (azn[pml] * ssxz_z[pml] + dt * c55[pml] * dz_vx[pml]) / azd[pml]
            sxx[pml] = ssxx_x[pml] + ssxx_z[pml]
            szz[pml] = sszz_x[pml] + sszz_z[pml]
            sxz[pml] = ssxz_x[pml] + ssxz_z[pml]

            step = iteration + 1
            if step % record_stride == 0:
                row = step // record_stride
                sx[row] = vx[case.rec_z, case.rec_x]
                sz[row] = vz[case.rec_z, case.rec_x]
                near_sx[row, 0] = vx[case.src_z, near_x]
                near_sz[row, 0] = vz[case.src_z, near_x]
            if step == snapshot_step:
                snapshot = {"snapshot_vx": vx.copy(), "snapshot_vz": vz.copy()}
        elapsed = time.perf_counter() - start
        if snapshot is None:
            raise RuntimeError("snapshot was not captured")
        diagnostics = {
            **snapshot,
            "near_Sx": near_sx,
            "near_Sz": near_sz,
            "snapshot_step": snapshot_step,
            "snapshot_time": (snapshot_step - 1) * case.dt,
            "src_x": case.src_x,
            "src_z": case.src_z,
            "near_rec_x": near_x,
            "near_rec_z": case.src_z,
            "record_stride": record_stride,
        }
        return {
            "Sx": sx,
            "Sz": sz,
            "runtime_s": elapsed,
            "diagnostics": diagnostics,
            "region_stats": self.stats,
            "kernel_design": "numpy-reference-splitpml-vti2d",
        }

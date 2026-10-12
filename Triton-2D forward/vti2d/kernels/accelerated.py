"""Fused interior and split-PML updates for 2-D VTI propagation.

The launchers pass region indices rather than scanning the full grid.
PML kernels combine derivatives, damped split updates, and reconstruction.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .common import bwd, fwd


@triton.jit
def interior_velocity_kernel(
    indices, npoints,
    vx, vz, sxx, szz, sxz, bx, bz,
    nx: tl.constexpr, nz: tl.constexpr,
    dx, dz, dt,
    c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
    c3: tl.constexpr, c4: tl.constexpr,
    HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    # Use stress derivatives to update velocities only at interior region indices.
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < npoints
    index = tl.load(indices + offset, mask=active, other=0)
    dx_sxx = fwd(sxx, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_sxz = bwd(sxz, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    dx_sxz = bwd(sxz, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_szz = fwd(szz, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    inverse_rho_x = tl.load(bx + index, mask=active, other=0.0)
    inverse_rho_z = tl.load(bz + index, mask=active, other=0.0)
    tl.store(
        vx + index,
        tl.load(vx + index, mask=active, other=0.0) + dt * inverse_rho_x * (dx_sxx + dz_sxz),
        mask=active,
    )
    tl.store(
        vz + index,
        tl.load(vz + index, mask=active, other=0.0) + dt * inverse_rho_z * (dx_sxz + dz_szz),
        mask=active,
    )


@triton.jit
def interior_stress_kernel(
    indices, npoints,
    vx, vz, sxx, szz, sxz,
    c11, c13, c33, c55,
    nx: tl.constexpr, nz: tl.constexpr,
    dx, dz, dt,
    c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
    c3: tl.constexpr, c4: tl.constexpr,
    HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    # Apply the VTI stiffness tensor to staggered velocity derivatives.
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < npoints
    index = tl.load(indices + offset, mask=active, other=0)
    dx_vx = bwd(vx, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_vz = bwd(vz, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    dx_vz = fwd(vz, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_vx = fwd(vx, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    local_c11 = tl.load(c11 + index, mask=active, other=0.0)
    local_c13 = tl.load(c13 + index, mask=active, other=0.0)
    local_c33 = tl.load(c33 + index, mask=active, other=0.0)
    local_c55 = tl.load(c55 + index, mask=active, other=0.0)
    tl.store(
        sxx + index,
        tl.load(sxx + index, mask=active, other=0.0) + dt * (local_c11 * dx_vx + local_c13 * dz_vz),
        mask=active,
    )
    tl.store(
        szz + index,
        tl.load(szz + index, mask=active, other=0.0) + dt * (local_c13 * dx_vx + local_c33 * dz_vz),
        mask=active,
    )
    tl.store(
        sxz + index,
        tl.load(sxz + index, mask=active, other=0.0) + dt * local_c55 * (dx_vz + dz_vx),
        mask=active,
    )


@triton.jit
def pml_velocity_kernel(
    indices, npoints,
    vx, vz, sxx, szz, sxz, bx, bz,
    svx_x, svx_z, svz_x, svz_z,
    pmlx, pmlz,
    nx: tl.constexpr, nz: tl.constexpr,
    dx, dz, dt,
    c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
    c3: tl.constexpr, c4: tl.constexpr,
    HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    # Advance directional velocity states and sum them to reconstruct Vx/Vz.
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < npoints
    index = tl.load(indices + offset, mask=active, other=0)
    dx_sxx = fwd(sxx, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_sxz = bwd(sxz, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    dx_sxz = bwd(sxz, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_szz = fwd(szz, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    inverse_rho_x = tl.load(bx + index, mask=active, other=0.0)
    inverse_rho_z = tl.load(bz + index, mask=active, other=0.0)
    damping_x = tl.load(pmlx + index, mask=active, other=0.0)
    damping_z = tl.load(pmlz + index, mask=active, other=0.0)
    axn, axd = 1.0 - 0.5 * dt * damping_x, 1.0 + 0.5 * dt * damping_x
    azn, azd = 1.0 - 0.5 * dt * damping_z, 1.0 + 0.5 * dt * damping_z
    value_vx_x = (axn * tl.load(svx_x + index, mask=active, other=0.0) + dt * inverse_rho_x * dx_sxx) / axd
    value_vx_z = (azn * tl.load(svx_z + index, mask=active, other=0.0) + dt * inverse_rho_x * dz_sxz) / azd
    value_vz_x = (axn * tl.load(svz_x + index, mask=active, other=0.0) + dt * inverse_rho_z * dx_sxz) / axd
    value_vz_z = (azn * tl.load(svz_z + index, mask=active, other=0.0) + dt * inverse_rho_z * dz_szz) / azd
    tl.store(svx_x + index, value_vx_x, mask=active)
    tl.store(svx_z + index, value_vx_z, mask=active)
    tl.store(svz_x + index, value_vz_x, mask=active)
    tl.store(svz_z + index, value_vz_z, mask=active)
    tl.store(vx + index, value_vx_x + value_vx_z, mask=active)
    tl.store(vz + index, value_vz_x + value_vz_z, mask=active)


@triton.jit
def pml_stress_kernel(
    indices, npoints,
    vx, vz, sxx, szz, sxz,
    c11, c13, c33, c55,
    ssxx_x, ssxx_z, sszz_x, sszz_z, ssxz_x, ssxz_z,
    pmlx, pmlz,
    nx: tl.constexpr, nz: tl.constexpr,
    dx, dz, dt,
    c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
    c3: tl.constexpr, c4: tl.constexpr,
    HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    # Advance directional stress states and reconstruct Sxx/Szz/Sxz.
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < npoints
    index = tl.load(indices + offset, mask=active, other=0)
    dx_vx = bwd(vx, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_vz = bwd(vz, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    dx_vz = fwd(vz, index, active, 1, 1.0 / dx, c0, c1, c2, c3, c4, HALF)
    dz_vx = fwd(vx, index, active, nx, 1.0 / dz, c0, c1, c2, c3, c4, HALF)
    local_c11 = tl.load(c11 + index, mask=active, other=0.0)
    local_c13 = tl.load(c13 + index, mask=active, other=0.0)
    local_c33 = tl.load(c33 + index, mask=active, other=0.0)
    local_c55 = tl.load(c55 + index, mask=active, other=0.0)
    damping_x = tl.load(pmlx + index, mask=active, other=0.0)
    damping_z = tl.load(pmlz + index, mask=active, other=0.0)
    axn, axd = 1.0 - 0.5 * dt * damping_x, 1.0 + 0.5 * dt * damping_x
    azn, azd = 1.0 - 0.5 * dt * damping_z, 1.0 + 0.5 * dt * damping_z
    value_sxx_x = (axn * tl.load(ssxx_x + index, mask=active, other=0.0) + dt * local_c11 * dx_vx) / axd
    value_sxx_z = (azn * tl.load(ssxx_z + index, mask=active, other=0.0) + dt * local_c13 * dz_vz) / azd
    value_szz_x = (axn * tl.load(sszz_x + index, mask=active, other=0.0) + dt * local_c13 * dx_vx) / axd
    value_szz_z = (azn * tl.load(sszz_z + index, mask=active, other=0.0) + dt * local_c33 * dz_vz) / azd
    value_sxz_x = (axn * tl.load(ssxz_x + index, mask=active, other=0.0) + dt * local_c55 * dx_vz) / axd
    value_sxz_z = (azn * tl.load(ssxz_z + index, mask=active, other=0.0) + dt * local_c55 * dz_vx) / azd
    tl.store(ssxx_x + index, value_sxx_x, mask=active)
    tl.store(ssxx_z + index, value_sxx_z, mask=active)
    tl.store(sszz_x + index, value_szz_x, mask=active)
    tl.store(sszz_z + index, value_szz_z, mask=active)
    tl.store(ssxz_x + index, value_sxz_x, mask=active)
    tl.store(ssxz_z + index, value_sxz_z, mask=active)
    tl.store(sxx + index, value_sxx_x + value_sxx_z, mask=active)
    tl.store(szz + index, value_szz_x + value_szz_z, mask=active)
    tl.store(sxz + index, value_sxz_x + value_sxz_z, mask=active)


def _grid(point_count: int, block: int):
    return (triton.cdiv(point_count, block),)


def launch_interior_velocity(indices, fields, inverse_density, shape, spacing, dt, coefficients, half, block):
    point_count = int(indices.numel())
    if point_count == 0:
        return
    nz, nx = shape
    dx, dz = spacing
    interior_velocity_kernel[_grid(point_count, block)](
        indices, point_count, *fields, *inverse_density,
        nx, nz, dx, dz, dt, *coefficients, HALF=half, BLOCK=block,
    )


def launch_interior_stress(indices, fields, stiffness, shape, spacing, dt, coefficients, half, block):
    point_count = int(indices.numel())
    if point_count == 0:
        return
    nz, nx = shape
    dx, dz = spacing
    interior_stress_kernel[_grid(point_count, block)](
        indices, point_count, *fields, *stiffness,
        nx, nz, dx, dz, dt, *coefficients, HALF=half, BLOCK=block,
    )


def launch_pml_velocity(indices, fields, inverse_density, split, pml, shape, spacing, dt, coefficients, half, block):
    point_count = int(indices.numel())
    if point_count == 0:
        return
    nz, nx = shape
    dx, dz = spacing
    pml_velocity_kernel[_grid(point_count, block)](
        indices, point_count, *fields, *inverse_density, *split, *pml,
        nx, nz, dx, dz, dt, *coefficients, HALF=half, BLOCK=block,
    )


def launch_pml_stress(indices, fields, stiffness, split, pml, shape, spacing, dt, coefficients, half, block):
    point_count = int(indices.numel())
    if point_count == 0:
        return
    nz, nx = shape
    dx, dz = spacing
    pml_stress_kernel[_grid(point_count, block)](
        indices, point_count, *fields, *stiffness, *split, *pml,
        nx, nz, dx, dz, dt, *coefficients, HALF=half, BLOCK=block,
    )

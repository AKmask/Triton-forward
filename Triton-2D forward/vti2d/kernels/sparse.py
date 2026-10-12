"""Inject a point stress source and sample 2-D receiver velocities.

Sources inside PML also update the split stresses so reconstruction
preserves the injected amplitude. Records use [time, receiver] order.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def inject_source_kernel(
    sxx, szz, sxz, source, iteration, index,
    weight_sxx: tl.constexpr, weight_szz: tl.constexpr, weight_sxz: tl.constexpr,
):
    # Add this time sample at the flattened source index in stress-component order.
    amplitude = tl.load(source + iteration)
    tl.store(sxx + index, tl.load(sxx + index) + amplitude * weight_sxx)
    tl.store(szz + index, tl.load(szz + index) + amplitude * weight_szz)
    tl.store(sxz + index, tl.load(sxz + index) + amplitude * weight_sxz)


@triton.jit
def inject_split_source_kernel(
    source, iteration, index,
    ssxx_x, ssxx_z, sszz_x, sszz_z, ssxz_x, ssxz_z,
    weight_sxx: tl.constexpr, weight_szz: tl.constexpr, weight_sxz: tl.constexpr,
):
    # Split each source amplitude equally between its x/z PML stress states.
    amplitude = tl.load(source + iteration)
    half_sxx = 0.5 * amplitude * weight_sxx
    half_szz = 0.5 * amplitude * weight_szz
    half_sxz = 0.5 * amplitude * weight_sxz
    tl.store(ssxx_x + index, tl.load(ssxx_x + index) + half_sxx)
    tl.store(ssxx_z + index, tl.load(ssxx_z + index) + half_sxx)
    tl.store(sszz_x + index, tl.load(sszz_x + index) + half_szz)
    tl.store(sszz_z + index, tl.load(sszz_z + index) + half_szz)
    tl.store(ssxz_x + index, tl.load(ssxz_x + index) + half_sxz)
    tl.store(ssxz_z + index, tl.load(ssxz_z + index) + half_sxz)


@triton.jit
def sample_kernel(vx, vz, receiver_indices, output_x, output_z, row, nrec, BLOCK: tl.constexpr):
    # Gather velocity samples into one row of each [time, receiver] output array.
    receiver_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = receiver_offset < nrec
    model_index = tl.load(receiver_indices + receiver_offset, mask=active, other=0)
    output_index = row * nrec + receiver_offset
    tl.store(output_x + output_index, tl.load(vx + model_index, mask=active, other=0.0), mask=active)
    tl.store(output_z + output_index, tl.load(vz + model_index, mask=active, other=0.0), mask=active)


def launch_source(stress_fields, source, iteration, index, weights):
    inject_source_kernel[(1,)](
        *stress_fields, source, int(iteration), int(index),
        *(float(value) for value in weights), num_warps=1,
    )


def launch_split_source(split_stress, source, iteration, index, weights):
    inject_split_source_kernel[(1,)](
        source, int(iteration), int(index), *split_stress,
        *(float(value) for value in weights), num_warps=1,
    )


def launch_sample(velocity_fields, receiver_indices, outputs, row, block=256):
    receiver_count = int(receiver_indices.numel())
    sample_kernel[(triton.cdiv(receiver_count, block),)](
        *velocity_fields, receiver_indices, *outputs, int(row), receiver_count, BLOCK=block,
    )

"""Masked staggered first derivatives shared by the 2-D Triton kernels.

The flattened strides are 1 for x and nx for z. HALF selects the stencil
radius at compilation; region indices must exclude the stencil halo.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fwd(
    pointer,
    index,
    active,
    stride: tl.constexpr,
    inverse_spacing,
    c0: tl.constexpr,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    HALF: tl.constexpr,
):
    # Evaluate the derivative at the forward half-grid position.
    result = c0 * (
        tl.load(pointer + index + stride, mask=active, other=0.0)
        - tl.load(pointer + index, mask=active, other=0.0)
    )
    if HALF >= 2:
        result += c1 * (
            tl.load(pointer + index + 2 * stride, mask=active, other=0.0)
            - tl.load(pointer + index - stride, mask=active, other=0.0)
        )
    if HALF >= 3:
        result += c2 * (
            tl.load(pointer + index + 3 * stride, mask=active, other=0.0)
            - tl.load(pointer + index - 2 * stride, mask=active, other=0.0)
        )
    if HALF >= 4:
        result += c3 * (
            tl.load(pointer + index + 4 * stride, mask=active, other=0.0)
            - tl.load(pointer + index - 3 * stride, mask=active, other=0.0)
        )
    if HALF >= 5:
        result += c4 * (
            tl.load(pointer + index + 5 * stride, mask=active, other=0.0)
            - tl.load(pointer + index - 4 * stride, mask=active, other=0.0)
        )
    return result * inverse_spacing


@triton.jit
def bwd(
    pointer,
    index,
    active,
    stride: tl.constexpr,
    inverse_spacing,
    c0: tl.constexpr,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    HALF: tl.constexpr,
):
    # Evaluate the derivative at the backward half-grid position.
    result = c0 * (
        tl.load(pointer + index, mask=active, other=0.0)
        - tl.load(pointer + index - stride, mask=active, other=0.0)
    )
    if HALF >= 2:
        result += c1 * (
            tl.load(pointer + index + stride, mask=active, other=0.0)
            - tl.load(pointer + index - 2 * stride, mask=active, other=0.0)
        )
    if HALF >= 3:
        result += c2 * (
            tl.load(pointer + index + 2 * stride, mask=active, other=0.0)
            - tl.load(pointer + index - 3 * stride, mask=active, other=0.0)
        )
    if HALF >= 4:
        result += c3 * (
            tl.load(pointer + index + 3 * stride, mask=active, other=0.0)
            - tl.load(pointer + index - 4 * stride, mask=active, other=0.0)
        )
    if HALF >= 5:
        result += c4 * (
            tl.load(pointer + index + 4 * stride, mask=active, other=0.0)
            - tl.load(pointer + index - 5 * stride, mask=active, other=0.0)
        )
    return result * inverse_spacing

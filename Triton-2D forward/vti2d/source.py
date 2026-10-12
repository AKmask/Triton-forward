"""Generate a sampled Ricker wavelet and 2-D stress-source weights.

Time is in seconds, frequency in hertz, and stress order is [sxx, szz, sxz].
"""

from __future__ import annotations

import math
import numpy as np


SOURCE_KINDS = ("explosive", "dc_xz")


def ricker(total_time: float, dt: float, f0: float, t0: float, scale: float = 1.0) -> dict:
    """Sample a delayed Ricker wavelet, including the time-zero sample."""
    count = int(math.floor(float(total_time) / float(dt) + 0.5)) + 1
    time = float(dt) * np.arange(count, dtype=np.float64)
    tau = time - float(t0)
    wavelet = float(scale) * (1.0 - 2.0 * (np.pi * float(f0) * tau) ** 2)
    wavelet *= np.exp(-(np.pi * float(f0) * tau) ** 2)
    return {"src": np.ascontiguousarray(wavelet), "t": time, "N": count}


def stress_source_weights(kind: str, dtype=np.float64) -> np.ndarray:
    """Return source weights in [sxx, szz, sxz] order."""
    normalized = str(kind).lower()
    if normalized == "explosive":
        weights = (-1.0, -1.0, 0.0)
    elif normalized == "dc_xz":
        weights = (0.0, 0.0, -math.sqrt(0.5))
    else:
        raise ValueError(f"source kind must be one of {SOURCE_KINDS}, got {kind!r}")
    return np.asarray(weights, dtype=dtype)

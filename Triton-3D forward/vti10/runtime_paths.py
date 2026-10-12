"""Keep forward-run caches and temporary files inside the upload directory."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


def configure_runtime_cache() -> Path:
    """Configure process-local cache paths before importing plotting/GPU libraries.

    Both forward packages share the .cache directory beside their project
    folders. No persistent user or system environment variables are changed.
    """
    cache_root = Path(__file__).resolve().parents[2] / ".cache"
    locations = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "PYTHONPYCACHEPREFIX": cache_root / "python",
        "TEMP": cache_root / "tmp",
        "TMP": cache_root / "tmp",
        "TMPDIR": cache_root / "tmp",
    }
    for variable, directory in locations.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    # These settings also cover imports and tempfile calls in this process.
    sys.pycache_prefix = str(locations["PYTHONPYCACHEPREFIX"])
    tempfile.tempdir = str(locations["TEMP"])
    return cache_root

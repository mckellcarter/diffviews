"""
UMAP/KNN backend abstraction — auto-detects cuML GPU vs umap-learn/sklearn CPU.

Environment:
    DIFFVIEWS_FORCE_CPU=1  — force CPU backend even if cuML available
"""
# pylint: disable=import-outside-toplevel

import os
from typing import Any

import numpy as np


def _gpu_available() -> bool:
    """Check if cuML GPU backend is available."""
    if os.environ.get("DIFFVIEWS_FORCE_CPU", "").lower() in ("1", "true"):
        return False
    try:
        import cuml  # noqa: F401
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


# Cached at import time
CUML_AVAILABLE = _gpu_available()


def get_umap_class() -> type:
    """Return UMAP class (cuML if GPU available, else umap-learn)."""
    if CUML_AVAILABLE:
        from cuml.manifold import UMAP
    else:
        from umap import UMAP
    return UMAP


def get_knn_class() -> type:
    """Return NearestNeighbors class (cuML if GPU, else sklearn)."""
    if CUML_AVAILABLE:
        from cuml.neighbors import NearestNeighbors
    else:
        from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors


def to_gpu_array(data: np.ndarray) -> Any:
    """Convert to cupy array if GPU available, else return numpy."""
    if CUML_AVAILABLE:
        import cupy as cp
        return cp.asarray(data)
    return np.asarray(data)


def to_numpy(data: Any) -> np.ndarray:
    """Ensure numpy array (convert from cupy if needed)."""
    if hasattr(data, "get"):
        return data.get()
    return np.asarray(data)


def get_backend_name() -> str:
    """Return current backend name for logging."""
    return "cuML (GPU)" if CUML_AVAILABLE else "umap-learn (CPU)"

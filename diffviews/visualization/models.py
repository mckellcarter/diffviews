"""
Model data container for diffviews visualization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelData:
    """Per-model data container for thread-safe multi-user access.

    All fields are populated during initialization and should be treated
    as read-only afterward to ensure thread safety.
    """
    name: str
    data_dir: Path
    adapter_name: str
    checkpoint_path: Optional[Path]
    noise_max: float  # Noise level 0-100 (100=pure noise)
    noise_min: float  # Noise level 0-100 (0=clean)
    default_steps: int
    default_guidance: float = 1.0

    # Loaded data
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    activations: Optional[np.ndarray] = None
    metadata_df: Optional[pd.DataFrame] = None
    umap_reducer: Any = None
    umap_scaler: Any = None
    umap_pca: Any = None  # PCA pre-reducer (if used)
    umap_params: Dict = field(default_factory=dict)
    nn_model: Any = None  # NearestNeighbors (sklearn or cuML)

    # Lazy-loaded adapter (protected by lock in visualizer)
    adapter: Any = None
    layer_shapes: Dict[str, tuple] = field(default_factory=dict)

    # Conditioning type: "class" for ImageNet models, "text" for T2I models
    conditioning_type: str = "class"

    # Default (pre-computed) embeddings backup for restore after layer change
    default_df: Optional[pd.DataFrame] = None
    default_activations: Optional[np.ndarray] = None
    default_umap_reducer: Any = None
    default_umap_scaler: Any = None
    default_umap_pca: Any = None
    default_umap_params: Optional[Dict] = None
    default_nn_model: Any = None  # NearestNeighbors (sklearn or cuML)
    current_layer: str = "default"

    # AlignedUMAP 3D mode fields
    is_3d_mode: bool = False
    sigma_levels: list = field(default_factory=list)
    embeddings_per_sigma: Dict[float, np.ndarray] = field(default_factory=dict)
    nn_models_per_sigma: Dict[float, Any] = field(default_factory=dict)
    umap_pkl_path: Optional[Path] = None  # For lazy loading

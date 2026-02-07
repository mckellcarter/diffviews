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
    sigma_max: float
    sigma_min: float
    default_steps: int

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

    # Default (pre-computed) embeddings backup for restore after layer change
    default_df: Optional[pd.DataFrame] = None
    default_activations: Optional[np.ndarray] = None
    default_umap_reducer: Any = None
    default_umap_scaler: Any = None
    default_umap_pca: Any = None
    default_umap_params: Optional[Dict] = None
    default_nn_model: Any = None  # NearestNeighbors (sklearn or cuML)
    current_layer: str = "default"

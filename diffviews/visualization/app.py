"""
Gradio-based diffusion activation visualizer.
Port of the Dash visualization app with multi-user support.
"""

import argparse
import json
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import plotly.graph_objects as go

from diffviews.processing.umap import load_dataset_activations
from diffviews.utils.device import get_device
from diffviews.adapters.registry import get_adapter
from diffviews.core.masking import ActivationMasker, unflatten_activation
from diffviews.core.generator import generate_with_mask_multistep


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
    umap_params: Dict = field(default_factory=dict)
    nn_model: Optional[NearestNeighbors] = None

    # Lazy-loaded adapter (protected by lock in visualizer)
    adapter: Any = None
    layer_shapes: Dict[str, tuple] = field(default_factory=dict)


class GradioVisualizer:
    """Gradio-based visualizer with multi-user support.

    Thread-safety model:
    - model_data dict is populated at init, read-only afterward
    - Each model's ModelData contains all model-specific state
    - current_model selection is per-session via gr.State (not instance attr)
    - Adapter loading protected by _generation_lock
    """

    def __init__(
        self,
        data_dir: Path,
        embeddings_path: Path = None,
        checkpoint_path: Path = None,
        device: str = "cuda",
        num_steps: int = 5,
        mask_steps: int = 1,
        guidance_scale: float = 1.0,
        sigma_max: float = 80.0,
        sigma_min: float = 0.5,
        label_dropout: float = 0.0,
        adapter_name: str = "dmd2-imagenet-64",
        max_classes: int = None,
        initial_model: str = None,
    ):
        """Initialize visualizer.

        Args:
            data_dir: Root data directory (parent containing model subdirs)
            embeddings_path: Optional path to precomputed embeddings CSV
            checkpoint_path: Optional path to checkpoint for generation
            device: Device for generation ('cuda', 'mps', or 'cpu')
            num_steps: Number of denoising steps
            mask_steps: Steps to apply activation mask
            guidance_scale: CFG scale
            sigma_max: Maximum sigma for denoising schedule
            sigma_min: Minimum sigma for denoising schedule
            label_dropout: Label dropout for model config
            adapter_name: Adapter name for model loading
            max_classes: Maximum number of classes to load (None=all)
            initial_model: Initial model to load
        """
        self.root_data_dir = Path(data_dir)
        self.device = device
        self.mask_steps = mask_steps
        self.guidance_scale = guidance_scale
        self.label_dropout = label_dropout
        self.max_classes = max_classes

        # Fallback defaults (used if no model config overrides)
        self._default_num_steps = num_steps
        self._default_sigma_max = sigma_max
        self._default_sigma_min = sigma_min
        self._default_adapter_name = adapter_name
        self._default_checkpoint_path = checkpoint_path
        self._default_embeddings_path = embeddings_path

        # Thread lock for adapter loading
        self._generation_lock = threading.Lock()

        # Shared class labels (typically same across models for ImageNet)
        self.class_labels: Dict[int, str] = {}
        self.load_class_labels()

        # Discover and load all models (read-only after init)
        self.model_configs: Dict[str, dict] = {}
        self.model_data: Dict[str, ModelData] = {}
        self.discover_models()
        self._load_all_models()

        # Determine default model (for initial UI state, not mutable)
        if initial_model and initial_model in self.model_data:
            self.default_model = initial_model
        elif "dmd2" in self.model_data:
            self.default_model = "dmd2"
        elif self.model_data:
            self.default_model = list(self.model_data.keys())[0]
        else:
            self.default_model = None

        if self.default_model:
            print(f"Default model: {self.default_model}")
            print(f"Loaded {len(self.model_data[self.default_model].df)} samples")

    def discover_models(self):
        """Discover available models in the data directory."""
        for subdir in self.root_data_dir.iterdir():
            if not subdir.is_dir():
                continue
            config_path = subdir / "config.json"
            embeddings_dir = subdir / "embeddings"
            if config_path.exists() and embeddings_dir.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                embeddings_files = list(embeddings_dir.glob("*.csv"))
                if not embeddings_files:
                    continue
                model_name = subdir.name
                self.model_configs[model_name] = {
                    "data_dir": subdir,
                    "adapter": config.get("adapter", "dmd2-imagenet-64"),
                    "checkpoint": config.get("checkpoint"),
                    "sigma_max": config.get("sigma_max", 80.0),
                    "sigma_min": config.get("sigma_min", 0.5),
                    "default_steps": config.get("default_steps", 5),
                    "embeddings_path": embeddings_files[0],
                }
                print(f"Discovered model: {model_name} (adapter={config.get('adapter')})")

    def _load_all_models(self):
        """Load data for all discovered models into model_data dict.

        This is called once during __init__ to preload all model data.
        After this, model_data is read-only for thread safety.
        """
        for model_name, config in self.model_configs.items():
            print(f"Loading model: {model_name}")
            model_data = self._load_model_data(model_name, config)
            if model_data is not None:
                self.model_data[model_name] = model_data

    def _load_model_data(self, model_name: str, config: dict) -> Optional[ModelData]:
        """Load all data for a single model into a ModelData instance."""
        data_dir = config["data_dir"]
        embeddings_path = config.get("embeddings_path")

        # Resolve checkpoint path
        checkpoint_path = None
        if config.get("checkpoint"):
            ckpt = Path(config["checkpoint"])
            if not ckpt.is_absolute():
                for base in [data_dir, self.root_data_dir, Path.cwd()]:
                    candidate = base / ckpt
                    if candidate.exists():
                        ckpt = candidate
                        break
            checkpoint_path = ckpt

        # Create ModelData instance
        model_data = ModelData(
            name=model_name,
            data_dir=data_dir,
            adapter_name=config.get("adapter", self._default_adapter_name),
            checkpoint_path=checkpoint_path,
            sigma_max=config.get("sigma_max", self._default_sigma_max),
            sigma_min=config.get("sigma_min", self._default_sigma_min),
            default_steps=config.get("default_steps", self._default_num_steps),
        )

        # Load embeddings
        if embeddings_path and Path(embeddings_path).exists():
            print(f"  Loading embeddings from {embeddings_path}")
            model_data.df = pd.read_csv(embeddings_path)

            # Load UMAP params
            param_path = Path(embeddings_path).with_suffix(".json")
            if param_path.exists():
                with open(param_path, "r", encoding="utf-8") as f:
                    model_data.umap_params = json.load(f)

            # Load UMAP model for inverse_transform (optional)
            umap_model_path = Path(embeddings_path).with_suffix(".pkl")
            if umap_model_path.exists():
                print(f"  Loading UMAP model from {umap_model_path}")
                with open(umap_model_path, "rb") as f:
                    umap_data = pickle.load(f)
                    model_data.umap_reducer = umap_data["reducer"]
                    model_data.umap_scaler = umap_data["scaler"]

            # Filter by max_classes if specified
            if self.max_classes is not None and "class_label" in model_data.df.columns:
                unique_classes = model_data.df["class_label"].unique()
                if len(unique_classes) > self.max_classes:
                    classes_to_keep = unique_classes[: self.max_classes]
                    original_count = len(model_data.df)
                    model_data.df = model_data.df[
                        model_data.df["class_label"].isin(classes_to_keep)
                    ].reset_index(drop=True)
                    print(
                        f"  Filtered to {self.max_classes} classes: "
                        f"{original_count} -> {len(model_data.df)} samples"
                    )

            # Load activations for generation
            model_data.activations, model_data.metadata_df = self._load_activations(
                data_dir, "imagenet_real"
            )

            # Fit KNN model
            self._fit_knn_model(model_data)

            print(f"  Loaded {len(model_data.df)} samples")
        else:
            print(f"  Warning: No embeddings found for {model_name}")
            model_data.df = pd.DataFrame()

        return model_data

    def _load_activations(
        self, data_dir: Path, model_type: str
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Load raw activations for generation."""
        activation_dir = data_dir / "activations" / model_type
        metadata_path = data_dir / "metadata" / model_type / "dataset_info.json"

        if not activation_dir.exists():
            print(f"  Warning: Activation dir not found: {activation_dir}")
            return None, None

        if not metadata_path.exists():
            print(f"  Warning: No metadata found: {metadata_path}")
            return None, None

        activations, metadata_df = load_dataset_activations(activation_dir, metadata_path)
        return activations, metadata_df

    def _fit_knn_model(self, model_data: ModelData):
        """Fit KNN model on UMAP coordinates for a model."""
        if model_data.df.empty or "umap_x" not in model_data.df.columns:
            return

        umap_coords = model_data.df[["umap_x", "umap_y"]].values
        print(f"  [KNN] Fitting on {umap_coords.shape[0]} points")
        model_data.nn_model = NearestNeighbors(n_neighbors=21, metric="euclidean")
        model_data.nn_model.fit(umap_coords)

    def get_model(self, model_name: str) -> Optional[ModelData]:
        """Get ModelData for a model name, or None if not found."""
        return self.model_data.get(model_name)

    def is_valid_model(self, model_name: str) -> bool:
        """Check if a model name is valid."""
        return model_name in self.model_data

    def load_class_labels(self):
        """Load ImageNet class labels (shared across models)."""
        search_paths = [
            self.root_data_dir / "imagenet_standard_class_index.json",
            self.root_data_dir / "imagenet_class_labels.json",
            Path(__file__).parent.parent / "data" / "imagenet_standard_class_index.json",
        ]

        for label_path in search_paths:
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    raw_labels = json.load(f)
                    self.class_labels = {int(k): v[1] for k, v in raw_labels.items()}
                print(f"Loaded {len(self.class_labels)} ImageNet class labels")
                return

        print("Warning: Class labels not found")
        self.class_labels = {}

    def get_class_name(self, class_id: int) -> str:
        """Get human-readable class name for a class ID."""
        if class_id in self.class_labels:
            return self.class_labels[class_id]
        return f"Unknown class {class_id}"

    def find_knn_neighbors(
        self, model_name: str, idx: int, k: int = 5, exclude_selected: bool = True
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a point.

        Args:
            model_name: Name of the model to use
            idx: Index of the query point
            k: Number of neighbors to return
            exclude_selected: If True, exclude the query point from results

        Returns:
            List of (neighbor_idx, distance) tuples sorted by distance
        """
        model_data = self.get_model(model_name)
        if model_data is None or model_data.nn_model is None:
            return []
        if idx >= len(model_data.df):
            return []

        # Query k+1 since the point itself is included
        query_k = k + 1 if exclude_selected else k
        query_point = model_data.df.iloc[idx][["umap_x", "umap_y"]].values.reshape(1, -1)
        distances, indices = model_data.nn_model.kneighbors(query_point, n_neighbors=query_k)

        results = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            if exclude_selected and neighbor_idx == idx:
                continue
            results.append((int(neighbor_idx), float(dist)))

        return results[:k]

    def get_image(self, model_name: str, image_path: str) -> Optional[np.ndarray]:
        """Load image as numpy array for gr.Image.

        Args:
            model_name: Name of the model (determines data directory)
            image_path: Relative path to image within model's data directory
        """
        model_data = self.get_model(model_name)
        if model_data is None:
            return None
        try:
            full_path = model_data.data_dir / image_path
            return np.array(Image.open(full_path))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    @staticmethod
    def create_composite_image(
        main_img: np.ndarray,
        inset_img: np.ndarray,
        inset_ratio: float = 0.25,
        margin: int = 2,
        border_width: int = 0
    ) -> np.ndarray:
        """Create composite image with inset in upper-left corner.

        Args:
            main_img: Main image (denoised output) as numpy array
            inset_img: Inset image (noised input) as numpy array
            inset_ratio: Size of inset relative to main image
            margin: Pixel margin from corner
            border_width: Width of black border around inset

        Returns:
            Composite image as numpy array
        """
        from PIL import ImageDraw

        main_pil = Image.fromarray(main_img)
        inset_pil = Image.fromarray(inset_img)

        # Resize inset
        main_size = main_pil.size
        inset_size = (int(main_size[0] * inset_ratio), int(main_size[1] * inset_ratio))
        inset_pil = inset_pil.resize(inset_size, Image.Resampling.LANCZOS)

        # Paste inset in upper-left corner
        inset_x, inset_y = margin, margin
        main_pil.paste(inset_pil, (inset_x, inset_y))

        # Draw black border around inset
        if border_width > 0:
            draw = ImageDraw.Draw(main_pil)
            x0, y0 = inset_x - border_width, inset_y - border_width
            x1, y1 = inset_x + inset_size[0], inset_y + inset_size[1]
            for i in range(border_width):
                draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline="black")

        return np.array(main_pil)

    def load_adapter(self, model_name: str):
        """Lazily load the generation adapter for a model.

        Args:
            model_name: Name of the model to load adapter for

        Returns:
            Loaded adapter or None if not found
        """
        model_data = self.get_model(model_name)
        if model_data is None:
            return None

        # Return cached adapter if already loaded
        if model_data.adapter is not None:
            return model_data.adapter

        if not model_data.checkpoint_path or not Path(model_data.checkpoint_path).exists():
            print(f"Error: checkpoint not found at {model_data.checkpoint_path}")
            return None

        print(f"Loading adapter: {model_data.adapter_name} from {model_data.checkpoint_path}")
        AdapterClass = get_adapter(model_data.adapter_name)
        model_data.adapter = AdapterClass.from_checkpoint(
            model_data.checkpoint_path,
            device=self.device,
            label_dropout=self.label_dropout,
        )
        # Cache layer shapes
        model_data.layer_shapes = model_data.adapter.get_layer_shapes()
        print(f"Adapter loaded. Layers: {list(model_data.layer_shapes.keys())}")
        return model_data.adapter

    def prepare_activation_dict(
        self, model_name: str, neighbor_indices: List[int]
    ) -> Optional[Dict[str, "torch.Tensor"]]:
        """Prepare activation dict for masked generation.

        Args:
            model_name: Name of the model to use
            neighbor_indices: List of indices to average activations from

        Returns:
            Dict mapping layer_name -> (1, C, H, W) tensor, or None if error
        """
        import torch

        model_data = self.get_model(model_name)
        if model_data is None or model_data.activations is None:
            return None
        if len(neighbor_indices) == 0:
            return None

        # Get layers from UMAP params (must match order used during embedding)
        layers = sorted(model_data.umap_params.get("layers", ["encoder_bottleneck", "midblock"]))
        if not model_data.layer_shapes:
            if model_data.adapter is None:
                self.load_adapter(model_name)
            if model_data.adapter is None:
                return None

        # Average neighbor activations in high-D space
        neighbor_acts = model_data.activations[neighbor_indices]  # (N, D)
        center_activation = np.mean(neighbor_acts, axis=0, keepdims=True)  # (1, D)

        # Split into per-layer activations (MUST be sorted order!)
        activation_dict = {}
        offset = 0
        for layer_name in layers:
            if layer_name not in model_data.layer_shapes:
                print(f"Warning: layer {layer_name} not in adapter shapes")
                continue
            shape = model_data.layer_shapes[layer_name]  # (C, H, W)
            size = int(np.prod(shape))
            layer_act_flat = center_activation[0, offset : offset + size]
            layer_act = unflatten_activation(
                torch.from_numpy(layer_act_flat).float(), shape
            )
            activation_dict[layer_name] = layer_act  # (1, C, H, W)
            offset += size

        return activation_dict

    def get_plot_dataframe(
        self,
        model_name: str,
        selected_idx: Optional[int] = None,
        manual_neighbors: Optional[List[int]] = None,
        knn_neighbors: Optional[List[int]] = None,
        highlighted_class: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get DataFrame for ScatterPlot with highlight column.

        Args:
            model_name: Name of the model to use
            selected_idx: Index of selected point
            manual_neighbors: List of manually selected neighbor indices
            knn_neighbors: List of KNN neighbor indices
            highlighted_class: Class ID to highlight
        """
        model_data = self.get_model(model_name)
        if model_data is None or model_data.df.empty:
            return pd.DataFrame(columns=["umap_x", "umap_y", "highlight", "sample_id"])

        df = model_data.df

        # Create copy with highlight column
        plot_df = df[["umap_x", "umap_y", "sample_id"]].copy()
        if "class_label" in df.columns:
            plot_df["class_label"] = df["class_label"].astype(str)

        # Add highlight column for visual distinction
        plot_df["highlight"] = "normal"

        manual_neighbors = manual_neighbors or []
        knn_neighbors = knn_neighbors or []

        # Mark class highlights
        if highlighted_class is not None and "class_label" in df.columns:
            class_mask = df["class_label"] == highlighted_class
            plot_df.loc[class_mask, "highlight"] = "class_highlight"

        # Mark neighbors
        for idx in knn_neighbors:
            if idx < len(plot_df):
                plot_df.loc[idx, "highlight"] = "knn_neighbor"

        for idx in manual_neighbors:
            if idx < len(plot_df):
                plot_df.loc[idx, "highlight"] = "manual_neighbor"

        # Mark selected (highest priority)
        if selected_idx is not None and selected_idx < len(plot_df):
            plot_df.loc[selected_idx, "highlight"] = "selected"

        return plot_df

    def get_class_options(self, model_name: str) -> List[Tuple[str, int]]:
        """Get class options for dropdown.

        Args:
            model_name: Name of the model to use
        """
        model_data = self.get_model(model_name)
        if model_data is None or model_data.df.empty:
            return []
        if "class_label" not in model_data.df.columns:
            return []

        unique_classes = model_data.df["class_label"].dropna().unique()
        unique_classes = sorted([int(c) for c in unique_classes])

        return [(f"{c}: {self.get_class_name(c)}", c) for c in unique_classes]

    def get_color_map(self, model_name: str) -> dict:
        """Generate color map for class labels using plasma colormap.

        Args:
            model_name: Name of the model to use
        """
        model_data = self.get_model(model_name)
        if model_data is None or model_data.df.empty:
            return {}
        if "class_label" not in model_data.df.columns:
            return {}

        import matplotlib.pyplot as plt
        unique_classes = model_data.df["class_label"].dropna().unique()
        unique_classes = sorted([int(c) for c in unique_classes])

        if not unique_classes:
            return {}

        cmap = plt.cm.plasma
        color_map = {}
        for i, cls in enumerate(unique_classes):
            rgba = cmap(i / max(len(unique_classes) - 1, 1))
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            color_map[str(cls)] = hex_color
        return color_map

    def create_umap_figure(
        self,
        model_name: str,
        selected_idx: Optional[int] = None,
        manual_neighbors: Optional[List[int]] = None,
        knn_neighbors: Optional[List[int]] = None,
        highlighted_class: Optional[int] = None,
        trajectory: Optional[List[Tuple[float, float, float]]] = None,
    ) -> go.Figure:
        """Create Plotly figure for UMAP scatter plot.

        Args:
            model_name: Name of the model to use
            selected_idx: Index of currently selected point
            manual_neighbors: List of manually selected neighbor indices
            knn_neighbors: List of KNN neighbor indices
            highlighted_class: Class ID to highlight
            trajectory: List of (x, y, sigma) tuples for denoising trajectory

        Returns:
            Plotly Figure object
        """
        model_data = self.get_model(model_name)
        if model_data is None or model_data.df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data loaded")
            return fig

        df = model_data.df
        manual_neighbors = manual_neighbors or []
        knn_neighbors = knn_neighbors or []

        # Get color map for classes
        color_map = self.get_color_map(model_name)

        # Build customdata with df indices for click handling
        customdata = list(range(len(df)))

        # Determine point colors based on class
        if "class_label" in df.columns:
            colors = [color_map.get(str(int(c)), "#888888") for c in df["class_label"]]
        else:
            colors = ["#1f77b4"] * len(df)

        # Calculate opacity based on sigma (high sigma = low alpha, log scale)
        if "conditioning_sigma" in df.columns:
            sigmas = df["conditioning_sigma"].values
            log_sigmas = np.log(sigmas + 1e-6)  # Avoid log(0)
            log_min, log_max = log_sigmas.min(), log_sigmas.max()
            if log_max > log_min:
                # Normalize: high sigma (high log) -> low alpha, low sigma -> high alpha
                normalized = (log_max - log_sigmas) / (log_max - log_min)
                opacities = 0.4 + 0.6 * normalized  # Range [0.4, 1.0]
            else:
                opacities = [0.7] * len(df)
            opacities = opacities.tolist()
        else:
            opacities = 0.7

        # Create main scatter trace
        fig = go.Figure()

        # Add main data points (use scattergl for better performance)
        fig.add_trace(go.Scatter(
            x=df["umap_x"].tolist(),
            y=df["umap_y"].tolist(),
            mode="markers",
            marker=dict(
                size=6,
                color=colors,
                opacity=opacities,
            ),
            customdata=customdata,
            hovertemplate="<b>%{customdata}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            name="samples",
        ))

        # Highlight class if specified (class color with black outline, sigma-based opacity)
        if highlighted_class is not None and "class_label" in df.columns:
            class_mask = df["class_label"] == highlighted_class
            class_indices = df[class_mask].index.tolist()
            if class_indices:
                class_color = color_map.get(str(int(highlighted_class)), "#888888")
                # Get opacities for highlighted class points
                if isinstance(opacities, list):
                    class_opacities = [opacities[i] for i in class_indices]
                else:
                    class_opacities = opacities
                fig.add_trace(go.Scatter(
                    x=df.loc[class_mask, "umap_x"].tolist(),
                    y=df.loc[class_mask, "umap_y"].tolist(),
                    mode="markers",
                    marker=dict(size=10, color=class_color, opacity=class_opacities, line=dict(width=1, color="black")),
                    customdata=[int(i) for i in class_indices],
                    hoverinfo="skip",
                    name="class_highlight",
                    showlegend=False,
                ))

        # Highlight KNN neighbors (green ring)
        if knn_neighbors:
            knn_df = df.iloc[knn_neighbors]
            fig.add_trace(go.Scatter(
                x=knn_df["umap_x"].tolist(),
                y=knn_df["umap_y"].tolist(),
                mode="markers",
                marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2, color="lime")),
                customdata=knn_neighbors,
                hoverinfo="skip",
                name="knn_neighbors",
                showlegend=False,
            ))

        # Highlight manual neighbors (blue ring)
        if manual_neighbors:
            man_df = df.iloc[manual_neighbors]
            fig.add_trace(go.Scatter(
                x=man_df["umap_x"].tolist(),
                y=man_df["umap_y"].tolist(),
                mode="markers",
                marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2, color="cyan")),
                customdata=manual_neighbors,
                hoverinfo="skip",
                name="manual_neighbors",
                showlegend=False,
            ))

        # Highlight selected point (red ring, highest priority)
        if selected_idx is not None and selected_idx < len(df):
            sel_row = df.iloc[selected_idx]
            fig.add_trace(go.Scatter(
                x=[float(sel_row["umap_x"])],
                y=[float(sel_row["umap_y"])],
                mode="markers",
                marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color="red")),
                customdata=[selected_idx],
                hoverinfo="skip",
                name="selected",
                showlegend=False,
            ))

        # Denoising trajectories (supports multiple)
        trajectories = trajectory if trajectory else []
        # Handle both old format (single trajectory) and new format (list of trajectories)
        if trajectories and isinstance(trajectories[0], tuple):
            trajectories = [trajectories]  # Wrap single trajectory in list

        for traj_idx, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue

            traj_x = [t[0] for t in traj]
            traj_y = [t[1] for t in traj]
            traj_sigma = [t[2] for t in traj]

            # Line trace for trajectory path
            fig.add_trace(go.Scatter(
                x=traj_x,
                y=traj_y,
                mode="lines",
                line=dict(color="lime", width=3, dash="dash"),
                hoverinfo="skip",
                name=f"trajectory_line_{traj_idx}",
                showlegend=False,
            ))

            # Markers (sigma labels reserved for hover preview)
            # Green gradient: light (start/noisy) -> dark (end/clean)
            fig.add_trace(go.Scatter(
                x=traj_x,
                y=traj_y,
                mode="markers",
                marker=dict(
                    size=10,
                    color=list(range(len(traj))),
                    colorscale=[[0, "#90EE90"], [1, "#228B22"]],  # lightgreen -> forestgreen
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=f"Traj {traj_idx + 1} Step %{{customdata}}<br>σ=%{{text:.1f}}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
                text=traj_sigma,
                customdata=list(range(1,len(traj)+1)),
                name=f"trajectory_{traj_idx}",
                showlegend=False,
            ))

            # Start marker (diamond)
            fig.add_trace(go.Scatter(
                x=[traj_x[0]],
                y=[traj_y[0]],
                mode="markers",
                marker=dict(symbol="diamond", size=14, color="lime", line=dict(width=1, color="white")),
                hovertemplate=f"Traj {traj_idx + 1} Start (σ=%.1f)<extra></extra>" % traj_sigma[0],
                name=f"traj_start_{traj_idx}",
                showlegend=False,
            ))

            # End marker (star)
            fig.add_trace(go.Scatter(
                x=[traj_x[-1]],
                y=[traj_y[-1]],
                mode="markers",
                marker=dict(symbol="star", size=18, color="#228B22", line=dict(width=1, color="white")),
                hovertemplate=f"Traj {traj_idx + 1} End (σ=%.1f)<extra></extra>" % traj_sigma[-1],
                name=f"traj_end_{traj_idx}",
                showlegend=False,
            ))

        fig.update_layout(
            title="Activation UMAP",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode="closest",
            template="plotly_white",
            showlegend=False,
            autosize=True,
            margin=dict(l=40, r=10, t=35, b=40),
        )

        return fig


# JavaScript for Plotly click and hover handling
# Cleaned up version: removed debug functions, simplified retry logic
PLOTLY_HANDLER_JS = r"""
// State
let clickBox = null;
let hoverBox = null;
let hoverTimeout = null;
let lastHoverKey = null;
let currentPlotDiv = null;
let initComplete = false;

// Find textbox inputs (Gradio 6 nesting)
function findTextboxes() {
    if (!clickBox) {
        const c = document.querySelector('#click-data-box');
        if (c) clickBox = c.querySelector('textarea') || c.querySelector('input');
    }
    if (!hoverBox) {
        const h = document.querySelector('#hover-data-box');
        if (h) hoverBox = h.querySelector('textarea') || h.querySelector('input');
    }
    return clickBox && hoverBox;
}

// Send data to Python via textbox
function sendData(box, data) {
    if (!box) return;
    box.value = JSON.stringify(data);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
}

// Click handler - immediate
function handlePlotlyClick(data) {
    if (!data?.points?.length) return;
    const point = data.points[0];
    sendData(clickBox, {
        pointIndex: point.customdata,
        x: point.x,
        y: point.y,
        curveNumber: point.curveNumber
    });
}

// Hover handler - debounced
function handlePlotlyHover(data) {
    if (!data?.points?.length) return;
    const point = data.points[0];
    const traceName = point.data.name || '';

    // Trajectory point hover
    const trajMatch = traceName.match(/^trajectory_(\d+)$/);
    if (trajMatch) {
        const trajIdx = parseInt(trajMatch[1]);
        const stepIdx = point.customdata;
        const hoverKey = `traj_${trajIdx}_${stepIdx}`;
        if (hoverKey === lastHoverKey) return;
        clearTimeout(hoverTimeout);
        hoverTimeout = setTimeout(() => {
            lastHoverKey = hoverKey;
            sendData(hoverBox, {
                type: 'trajectory',
                trajIdx: trajIdx,
                stepIdx: stepIdx,
                x: point.x,
                y: point.y,
                sigma: point.text
            });
        }, 100);
        return;
    }

    // Only main data trace (curve 0)
    if (point.curveNumber !== 0) return;
    const idx = point.customdata;
    const hoverKey = `sample_${idx}`;
    if (hoverKey === lastHoverKey) return;
    clearTimeout(hoverTimeout);
    hoverTimeout = setTimeout(() => {
        lastHoverKey = hoverKey;
        sendData(hoverBox, {
            type: 'sample',
            pointIndex: idx,
            x: point.x,
            y: point.y
        });
    }, 100);
}

// Check if Plotly is ready
function isPlotlyReady(div) {
    return div && typeof div.on === 'function' && div.data && div.layout;
}

// Find Plotly div
function findPlotDiv() {
    return document.querySelector('#umap-plot .plotly-graph-div') ||
           document.querySelector('#umap-plot .js-plotly-plot') ||
           document.querySelector('.plotly-graph-div');
}

// Attach handlers (retries indefinitely until success)
function attachPlotlyHandlers() {
    const plotDiv = findPlotDiv();
    if (!plotDiv || !isPlotlyReady(plotDiv) || !findTextboxes()) {
        setTimeout(attachPlotlyHandlers, 300);
        return;
    }

    // Skip if already attached to this element
    if (plotDiv === currentPlotDiv && plotDiv._handlersAttached) return;

    // Clear existing handlers
    try {
        plotDiv.removeAllListeners('plotly_click');
        plotDiv.removeAllListeners('plotly_hover');
    } catch(e) {}

    // Attach
    currentPlotDiv = plotDiv;
    plotDiv._handlersAttached = true;
    initComplete = true;
    plotDiv.on('plotly_click', handlePlotlyClick);
    plotDiv.on('plotly_hover', handlePlotlyHover);
}

// MutationObserver for Gradio DOM replacement
function setupObserver() {
    const container = document.querySelector('#umap-plot');
    if (!container) {
        setTimeout(setupObserver, 500);
        return;
    }
    new MutationObserver(() => {
        if (initComplete) setTimeout(attachPlotlyHandlers, 100);
    }).observe(container, { childList: true, subtree: true });
}

// Initialize
setTimeout(() => {
    attachPlotlyHandlers();
    setupObserver();
}, 1500);

// Polling backup (1s interval after init)
setInterval(() => {
    if (!initComplete) return;
    const plotDiv = findPlotDiv();
    if (plotDiv && isPlotlyReady(plotDiv) && (plotDiv !== currentPlotDiv || !plotDiv._handlersAttached)) {
        attachPlotlyHandlers();
    }
}, 1000);
"""

# CSS for layout, plot sizing, and reduced chrome
# Module-level for Gradio 6 compatibility (passed to launch() not Blocks())
CUSTOM_CSS = """
    /* Main container fills viewport with min dimensions */
    .gradio-container {
        max-width: 100% !important;
        padding: 0.5rem !important;
        min-width: 900px !important;
        min-height: 600px !important;
        overflow: auto !important;
    }

    /* Main row - fixed height to prevent iframe expansion issues */
    #main-row {
        height: 1200px !important;
        max-height: 1200px !important;
        align-items: flex-start !important;
        flex-wrap: nowrap !important;
    }

    /* Sidebars: fixed width, scrollable */
    #left-sidebar, #right-sidebar {
        flex: 0 0 220px !important;
        max-height: 1200px !important;
        overflow-y: auto !important;
        padding: 0.25rem !important;
    }

    /* Center column stretches to fill remaining space */
    #center-column {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        min-width: 600px !important;
        max-height: 1200px !important;
    }

    /* Plot container - fixed height, no vh units (iframe-safe) */
    #umap-plot {
        height: 1000px !important;
        max-height: 1000px !important;
        min-height: 400px !important;
    }

    /* Make Plotly fill its container with constrained height */
    #umap-plot > div,
    #umap-plot .js-plotly-plot,
    #umap-plot .plotly-graph-div {
        height: 100% !important;
        max-height: 1000px !important;
        width: 100% !important;
    }

    /* Hidden textboxes for JS bridge */
    #click-data-box,
    div:has(> #click-data-box) {
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    #hover-data-box,
    div:has(> #hover-data-box) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    /* Reduce group padding */
    .gr-group {
        padding: 0.5rem !important;
        margin-bottom: 0.10rem !important;
    }

    /* Compact markdown headers */
    .gr-group h3 {
        margin: 0 0 0.25rem 0 !important;
        font-size: 0.9rem !important;
    }

    /* Smaller image containers */
    .gr-image {
        margin: 0 !important;
    }

    /* Compact sliders and inputs */
    .gr-slider, .gr-number {
        margin-bottom: 0.25rem !important;
    }

    /* Compact buttons */
    .gr-button-sm {
        padding: 0.25rem 0.5rem !important;
        margin: 0.125rem 0 !important;
    }

    /* Gallery compact */
    .gr-gallery {
        margin: 0.25rem 0 !important;
    }

    /* Reduce title size */
    h1 {
        font-size: 1.5rem !important;
        margin: 0.25rem 0 0.5rem 0 !important;
    }

    /* Model selector row: label + dropdown inline */
    #model-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.5rem !important;
        margin-bottom: 0.25rem !important;
    }

    #model-row > div {
        flex: 0 0 auto !important;
    }

    #model-label {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: 60px !important;
    }

    #model-label p {
        margin: 0 !important;
        font-size: 0.9rem !important;
    }

    #model-dropdown {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }

    /* KNN row styling */
    #knn-row {
        flex-wrap: nowrap !important;
        align-items: center !important;
    }

    #knn-label {
        flex-shrink: 0 !important;
    }

    #knn-label p {
        margin: 0 !important;
        white-space: nowrap !important;
    }

    #knn-input {
        flex-shrink: 0 !important;
        max-width: 65px !important;
    }

    /* Hide spin buttons on K input */
    #knn-input input[type="number"]::-webkit-inner-spin-button,
    #knn-input input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }

    #knn-input input[type="number"] {
        -moz-appearance: textfield !important;
    }

    #status-text {
        font-size: 0.8rem !important;
        color: #666 !important;
        margin: 0 0 0.5rem 0 !important;
    }

    #status-text p {
        margin: 0 !important;
    }

    /* Compact generation params - all in one row with inline labels */
    #gen-params-row {
        gap: 0.25rem !important;
        align-items: center !important;
        flex-wrap: wrap !important;
    }

    #gen-params-row > div {
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.2rem !important;
        flex: 0 1 auto !important;
    }

    #gen-params-row label {
        min-width: fit-content !important;
        margin: 0 !important;
        font-size: 0.75rem !important;
        white-space: nowrap !important;
    }

    #gen-params-row input {
        max-width: 50px !important;
        padding: 0.2rem 0.4rem !important;
        font-size: 0.85rem !important;
    }

    /* Hide number input spin buttons */
    #gen-params-row input[type="number"]::-webkit-inner-spin-button,
    #gen-params-row input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }

    #gen-params-row input[type="number"] {
        -moz-appearance: textfield !important;
    }

    /* Smooth scaling for 64x64 images (auto = bilinear interpolation) */
    #preview-image img,
    #selected-image img,
    #generated-image img,
    #intermediate-gallery img,
    #neighbor-gallery img {
        image-rendering: auto !important;
    }

    /* Image containers: fill available space */
    #preview-image, #selected-image, #generated-image {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
    }

    /* Preview image: force 300px height, preserve aspect ratio */
    #preview-image {
        width: 100% !important;
        min-height: 300px !important;
    }

    #preview-image img {
        height: 300px !important;
        width: auto !important;
        object-fit: contain !important;
    }

    /* Compact preview and selected details text */
    #preview-details,
    #selected-details {
        line-height: 1.3 !important;
    }

    #preview-details p,
    #selected-details p {
        margin: 0.15em 0 !important;
    }

    /* Selected/generated images: fill container */
    #selected-image img, #generated-image img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
        max-width: none !important;
        max-height: none !important;
    }

    /* Generated image: match preview size */
    #generated-image {
        width: 100% !important;
        min-height: 180px !important;
    }

    #generated-image img {
        min-height: 300px !important;
    }

    /* Gallery images also scale up */
    #intermediate-gallery img,
    #neighbor-gallery img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
    }

    /* Neighbor gallery: scrollable with more height */
    #neighbor-gallery {
        max-height: 280px !important;
        overflow-y: auto !important;
    }

    /* Lock scroll when gallery preview is active */
    #neighbor-gallery:has(.preview) {
        overflow-y: hidden !important;
    }

    /* Make preview fill the container */
    #neighbor-gallery .preview {
        max-height: 280px !important;
    }
"""


def create_gradio_app(visualizer: GradioVisualizer) -> gr.Blocks:
    """Create Gradio Blocks app.

    Note: In Gradio 6, theme/css/head are passed to launch() not Blocks().
    Use CUSTOM_CSS and PLOTLY_HANDLER_JS module constants when calling launch().
    """
    with gr.Blocks(
        title="Diffusion Activation Visualizer",
    ) as app:

        # Per-session state
        current_model = gr.State(value=visualizer.default_model)  # Model selection per session
        selected_idx = gr.State(value=None)
        manual_neighbors = gr.State(value=[])
        knn_neighbors = gr.State(value=[])
        knn_distances = gr.State(value={})  # {idx: distance} for KNN neighbors
        highlighted_class = gr.State(value=None)
        trajectory_coords = gr.State(value=[])  # [[(x, y, sigma), ...], ...] list of trajectories
        intermediate_images = gr.State(value=[])  # [(img, sigma), ...] for trajectory hover/animation
        animation_frame = gr.State(value=-1)  # Current animation frame (-1 = showing final)
        generation_info = gr.State(value=None)  # {class_id, class_name, n_traj} for display

        # Get initial model data for default values
        default_model_data = visualizer.get_model(visualizer.default_model)
        initial_sample_count = len(default_model_data.df) if default_model_data else 0

        gr.Markdown("# Diffusion Activation Visualizer")

        with gr.Row(elem_id="main-row"):
            # Left column (sidebar)
            with gr.Column(scale=1, elem_id="left-sidebar"):
                # Model selector + status
                with gr.Row(elem_id="model-row"):
                    gr.Markdown("**Model**", elem_id="model-label")
                    if len(visualizer.model_configs) > 1:
                        model_dropdown = gr.Dropdown(
                            choices=list(visualizer.model_configs.keys()),
                            value=visualizer.default_model,
                            show_label=False,
                            interactive=True,
                            elem_id="model-dropdown",
                        )
                    else:
                        model_dropdown = gr.Dropdown(
                            choices=[visualizer.default_model] if visualizer.default_model else [],
                            value=visualizer.default_model,
                            show_label=False,
                            visible=len(visualizer.model_configs) > 0,
                            elem_id="model-dropdown",
                        )
                status_text = gr.Markdown(
                    f"Showing {initial_sample_count} samples"
                    + (f" ({visualizer.default_model})" if visualizer.default_model else ""),
                    elem_id="status-text"
                )
                model_status = gr.Markdown("", visible=False)

                # Preview section (updated on hover)
                with gr.Group():
                    gr.Markdown("### Preview")
                    preview_image = gr.Image(
                        label=None, show_label=False, elem_id="preview-image"
                    )
                    preview_details = gr.Markdown(
                        "Hover over a point to preview", elem_id="preview-details"
                    )

                # Class filter
                with gr.Group():
                    gr.Markdown("### Class Filter")
                    class_dropdown = gr.Dropdown(
                        choices=visualizer.get_class_options(visualizer.default_model) if visualizer.default_model else [],
                        label="Select class",
                        interactive=True,
                    )
                    clear_class_btn = gr.Button("Clear Highlight", size="sm")
                    class_status = gr.Markdown("")

                # Selected sample (moved from right sidebar)
                with gr.Group():
                    gr.Markdown("### Selected Sample")
                    selected_image = gr.Image(
                        label=None, show_label=False, height=150, elem_id="selected-image"
                    )
                    selected_details = gr.Markdown(
                        "Click a point to select", elem_id="selected-details"
                    )
                    clear_selection_btn = gr.Button("Clear Selection", size="sm")

            # Center column (main plot)
            with gr.Column(scale=3, min_width=500, elem_id="center-column"):
                # Use Plotly via gr.Plot for proper click handling
                umap_plot = gr.Plot(
                    value=visualizer.create_umap_figure(visualizer.default_model) if visualizer.default_model else None,
                    elem_id="umap-plot",
                    show_label=False,
                )
                # Hidden textboxes for JS bridge (after plot to not affect top alignment)
                # Note: visible=True but hidden via CSS - Gradio 6 doesn't render visible=False
                click_data_box = gr.Textbox(
                    value="",
                    elem_id="click-data-box",
                    visible=True,  # Hidden via CSS, must be in DOM for JS bridge
                )
                hover_data_box = gr.Textbox(
                    value="",
                    elem_id="hover-data-box",
                    visible=True,  # Hidden via CSS, must be in DOM for JS bridge
                )

            # Right column (generation & neighbors)
            with gr.Column(scale=1, elem_id="right-sidebar"):
                # Generation settings
                with gr.Group(elem_id="gen-group"):
                    gr.Markdown("### Generation")
                    # Parameters and buttons first (use model defaults)
                    default_steps = default_model_data.default_steps if default_model_data else 5
                    default_sigma_max = default_model_data.sigma_max if default_model_data else 80.0
                    default_sigma_min = default_model_data.sigma_min if default_model_data else 0.5
                    with gr.Row(elem_id="gen-params-row"):
                        num_steps_slider = gr.Number(
                            value=default_steps, label="Steps",
                            elem_id="num-steps", min_width=50, precision=0
                        )
                        mask_steps_slider = gr.Number(
                            value=visualizer.mask_steps, label="Mask",
                            elem_id="mask-steps", min_width=50, precision=0
                        )
                        guidance_slider = gr.Number(
                            value=visualizer.guidance_scale, label="CFG",
                            elem_id="guidance", min_width=50
                        )
                        sigma_max_input = gr.Number(
                            value=default_sigma_max, label="σ max",
                            elem_id="sigma-max", min_width=50
                        )
                        sigma_min_input = gr.Number(
                            value=default_sigma_min, label="σ min",
                            elem_id="sigma-min", min_width=50
                        )
                    generate_btn = gr.Button("Generate from Neighbors", variant="primary")
                    gen_status = gr.Markdown("Select neighbors, then generate")
                    # Generated output
                    generated_image = gr.Image(
                        label=None, show_label=False, elem_id="generated-image"
                    )
                    # Denoising steps gallery with frame nav
                    intermediate_gallery = gr.Gallery(
                        label="Denoising Steps",
                        show_label=True,
                        columns=5,
                        rows=1,
                        height=70,
                        object_fit="contain",
                        buttons=[],  # Hide download/share buttons
                        elem_id="intermediate-gallery",
                    )
                    with gr.Row():
                        prev_frame_btn = gr.Button("◀", size="sm", min_width=40)
                        next_frame_btn = gr.Button("▶", size="sm", min_width=40)
                    clear_gen_btn = gr.Button("Clear", size="sm")

                # Neighbor list
                with gr.Group():
                    gr.Markdown("### Neighbors")
                    with gr.Row(elem_id="knn-row"):
                        gr.Markdown("**K-neighbors**", elem_id="knn-label")
                        knn_k_slider = gr.Number(
                            value=5, show_label=False, precision=0, minimum=1,
                            elem_id="knn-input", min_width=50
                        )
                        suggest_btn = gr.Button("Suggest KNN", size="sm")
                    neighbor_gallery = gr.Gallery(
                        label=None,
                        show_label=False,
                        columns=2,
                        height="auto",
                        object_fit="contain",
                        allow_preview=True,
                        elem_id="neighbor-gallery",
                    )
                    neighbor_info = gr.Markdown("No neighbors selected")
                    clear_neighbors_btn = gr.Button("Clear Neighbors", size="sm")

        # --- Event Handlers ---

        def on_load():
            """No-op - KNN models are already fit during init."""
            pass

        def build_neighbor_gallery(model_name, sel_idx, man_n, knn_n, knn_dist):
            """Build neighbor gallery and info text."""
            model_data = visualizer.get_model(model_name)
            if sel_idx is None or model_data is None:
                return [], "No neighbors selected"

            man_n = man_n or []
            knn_n = knn_n or []
            knn_dist = knn_dist or {}

            # Combine neighbors: KNN first (sorted by distance), then manual
            all_neighbors = []
            knn_with_dist = [(idx, knn_dist.get(idx, 999)) for idx in knn_n if idx not in man_n]
            knn_with_dist.sort(key=lambda x: x[1])
            all_neighbors.extend([idx for idx, _ in knn_with_dist])
            all_neighbors.extend(man_n)

            if not all_neighbors:
                return [], "Click points or use Suggest"

            images = []
            for idx in all_neighbors[:20]:
                if idx < len(model_data.df):
                    sample = model_data.df.iloc[idx]
                    img = visualizer.get_image(model_name, sample["image_path"])
                    if img is not None:
                        if "class_label" in sample:
                            cls_id = int(sample["class_label"])
                            cls_name = visualizer.get_class_name(cls_id)
                            label = f"{cls_id}: {cls_name}"
                        else:
                            label = f"#{idx}"
                        if idx in knn_dist:
                            label += f" (d={knn_dist[idx]:.2f})"
                        elif idx in man_n:
                            label += " (manual)"
                        images.append((img, label))

            knn_count = len([n for n in knn_n if n not in man_n])
            man_count = len(man_n)
            info = f"{len(all_neighbors)} neighbors"
            if knn_count > 0:
                info += f" ({knn_count} KNN"
            if man_count > 0:
                info += f", {man_count} manual" if knn_count > 0 else f" ({man_count} manual"
            if knn_count > 0 or man_count > 0:
                info += ")"
            return images, info

        def on_hover_data(hover_json, intermediates, model_name):
            """Handle plot hover via JS bridge - update preview panel.

            Handles two types:
            - Sample hover: show sample image from dataset
            - Trajectory hover: show intermediate image from generation
            """
            if not hover_json:
                return gr.update(), gr.update()

            try:
                hover_data = json.loads(hover_json)
            except (json.JSONDecodeError, TypeError, ValueError):
                return gr.update(), gr.update()

            hover_type = hover_data.get("type", "sample")

            # Trajectory point hover - show intermediate image
            if hover_type == "trajectory":
                step_idx = hover_data.get("stepIdx")
                traj_idx = hover_data.get("trajIdx", 0)
                sigma = hover_data.get("sigma", "?")

                if intermediates and step_idx is not None:
                    step_idx = int(step_idx)
                    if 0 <= step_idx < len(intermediates):
                        img, stored_sigma = intermediates[step_idx]
                        details = f"**Trajectory {traj_idx + 1}, Step {step_idx + 1}**\n\n"
                        details += f"σ = {stored_sigma:.1f}\n\n"
                        details += f"Coords: ({hover_data.get('x', 0):.2f}, {hover_data.get('y', 0):.2f})"
                        return img, details

                # No intermediates stored
                details = f"**Trajectory step {step_idx}**\n\nσ = {sigma}"
                return gr.update(), details

            # Sample hover - show dataset image
            model_data = visualizer.get_model(model_name)
            if model_data is None:
                return gr.update(), gr.update()

            point_idx = hover_data.get("pointIndex")
            if point_idx is None:
                return gr.update(), gr.update()
            point_idx = int(point_idx)

            if point_idx < 0 or point_idx >= len(model_data.df):
                return gr.update(), gr.update()

            sample = model_data.df.iloc[point_idx]
            img = visualizer.get_image(model_name, sample["image_path"])

            # Format details
            if "class_label" in sample:
                class_name = visualizer.get_class_name(int(sample["class_label"]))
            else:
                class_name = "N/A"
            details = f"**{sample['sample_id']}**<br>"
            if "class_label" in sample:
                details += f"Class: {int(sample['class_label'])}: {class_name}<br>"
            if "conditioning_sigma" in sample:
                details += f"σ = {sample['conditioning_sigma']:.1f}<br>"
            details += f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"

            return img, details

        def on_click_data(click_json, sel_idx, man_n, knn_n, knn_dist, high_class, traj, model_name):
            """Handle plot click via JS bridge - select point or toggle neighbor."""
            if not click_json:
                return (gr.update(),) * 9

            model_data = visualizer.get_model(model_name)
            if model_data is None:
                return (gr.update(),) * 9

            try:
                click_data = json.loads(click_json)
                # Only handle clicks on main samples trace (curve 0)
                # Ignore trajectory and other overlay traces
                if click_data.get("curveNumber", 0) != 0:
                    return (gr.update(),) * 9
                point_idx = click_data.get("pointIndex")
                if point_idx is None:
                    return (gr.update(),) * 9
                point_idx = int(point_idx)
            except (json.JSONDecodeError, TypeError, ValueError):
                return (gr.update(),) * 9

            if point_idx < 0 or point_idx >= len(model_data.df):
                return (gr.update(),) * 9

            knn_dist = knn_dist or {}

            # First click: select this point
            if sel_idx is None:
                sample = model_data.df.iloc[point_idx]
                img = visualizer.get_image(model_name, sample["image_path"])

                # Format details
                if "class_label" in sample:
                    class_name = visualizer.get_class_name(int(sample["class_label"]))
                else:
                    class_name = "N/A"
                details = f"**{sample['sample_id']}**<br>"
                if "class_label" in sample:
                    details += f"Class: {int(sample['class_label'])}: {class_name}<br>"
                if "conditioning_sigma" in sample:
                    details += f"σ = {sample['conditioning_sigma']:.1f}<br>"
                details += f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"

                # Build updated Plotly figure with selection (preserve trajectory)
                fig = visualizer.create_umap_figure(
                    model_name,
                    selected_idx=point_idx,
                    highlighted_class=high_class,
                    trajectory=traj if traj else None,
                )

                return (
                    img,           # selected_image
                    details,       # selected_details
                    point_idx,     # selected_idx
                    [],            # manual_neighbors
                    [],            # knn_neighbors
                    fig,           # umap_plot
                    [],            # neighbor_gallery
                    "Click points or use Suggest",  # neighbor_info
                    traj,          # trajectory_coords (preserved)
                )

            # Clicking same point: do nothing
            if point_idx == sel_idx:
                return (gr.update(),) * 9

            # Toggle neighbor (preserve trajectory)
            man_n = list(man_n) if man_n else []
            knn_n = list(knn_n) if knn_n else []
            at_limit = False

            if point_idx in man_n:
                man_n.remove(point_idx)
            elif point_idx in knn_n:
                knn_n.remove(point_idx)
            else:
                # Check total neighbor limit
                total = len(man_n) + len(knn_n)
                if total >= 20:
                    at_limit = True
                else:
                    man_n.append(point_idx)

            # Rebuild Plotly figure with updated highlights (preserve trajectory)
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )

            # Build gallery for neighbors
            gallery, info = build_neighbor_gallery(model_name, sel_idx, man_n, knn_n, knn_dist)

            # Add limit notice if needed
            if at_limit:
                info += " (max 20)"

            return (
                gr.update(),   # selected_image
                gr.update(),   # selected_details
                sel_idx,       # selected_idx (unchanged)
                man_n,         # manual_neighbors
                knn_n,         # knn_neighbors
                fig,           # umap_plot
                gallery,       # neighbor_gallery
                info,          # neighbor_info
                traj,          # trajectory_coords (preserved)
            )

        def on_clear_selection(high_class, traj, model_name):
            """Clear selection and neighbors (preserves trajectory, preview unchanged)."""
            fig = visualizer.create_umap_figure(
                model_name,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )
            return (
                None,                      # selected_image
                "Click a point to select", # selected_details
                None,                      # selected_idx
                [],                        # manual_neighbors
                [],                        # knn_neighbors
                {},                        # knn_distances
                fig,                       # umap_plot
                [],                        # neighbor_gallery
                "No neighbors selected",   # neighbor_info
            )

        def on_class_filter(class_value, sel_idx, man_n, knn_n, traj, model_name):
            """Handle class filter selection (preserves trajectory)."""
            model_data = visualizer.get_model(model_name)
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=class_value,
                trajectory=traj if traj else None,
            )

            if class_value is not None and model_data and "class_label" in model_data.df.columns:
                count = (model_data.df["class_label"] == class_value).sum()
                status = f"{count} samples"
            else:
                status = ""

            return fig, class_value, status

        def on_clear_class(sel_idx, man_n, knn_n, traj, model_name):
            """Clear class highlight (preserves trajectory)."""
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                trajectory=traj if traj else None,
            )
            return fig, None, None, ""

        def on_model_switch(new_model_name, cur_model, _sel_idx, _man_n, _knn_n, _knn_dist, _high_class):
            """Handle model switching (resets all state including preview).

            Note: This now just updates the session's current_model state.
            All model data is preloaded, so no shared state is modified.
            """
            if new_model_name == cur_model:
                return (gr.update(),) * 23  # +1 for current_model state

            if not visualizer.is_valid_model(new_model_name):
                return (gr.update(),) * 23

            model_data = visualizer.get_model(new_model_name)
            fig = visualizer.create_umap_figure(new_model_name)
            status = f"Showing {len(model_data.df)} samples ({new_model_name})"

            return (
                new_model_name,                    # current_model (session state)
                fig,                               # umap_plot
                status,                            # status_text
                f"Switched to {new_model_name}",   # model_status
                None,                              # selected_idx
                [],                                # manual_neighbors
                [],                                # knn_neighbors
                {},                                # knn_distances
                None,                              # highlighted_class
                [],                                # trajectory_coords
                None,                              # preview_image
                "Hover over a point to preview",   # preview_details
                None,                              # selected_image
                "Click a point to select",         # selected_details
                gr.update(choices=visualizer.get_class_options(new_model_name), value=None),  # class_dropdown
                None,                              # generated_image
                gr.update(value=[], label="Denoising Steps"),  # intermediate_gallery
                "Select neighbors, then generate", # gen_status
                [],                                # intermediate_images
                -1,                                # animation_frame
                None,                              # generation_info
                [],                                # neighbor_gallery
                "No neighbors selected",           # neighbor_info
            )

        # Wire up events
        # on_load initializes KNN model
        app.load(on_load, outputs=[])

        # Hover handling via JS bridge (updates preview panel)
        # Gradio 6: use .change() instead of .input()
        hover_data_box.change(
            on_hover_data,
            inputs=[hover_data_box, intermediate_images, current_model],
            outputs=[preview_image, preview_details],
        )

        # Click handling via JS bridge (click_data_box receives JSON from Plotly click)
        # Gradio 6: use .change() instead of .input()
        click_data_box.change(
            on_click_data,
            inputs=[
                click_data_box, selected_idx, manual_neighbors,
                knn_neighbors, knn_distances, highlighted_class, trajectory_coords,
                current_model
            ],
            outputs=[
                selected_image,
                selected_details,
                selected_idx,
                manual_neighbors,
                knn_neighbors,
                umap_plot,
                neighbor_gallery,
                neighbor_info,
                trajectory_coords,
            ],
        )

        clear_selection_btn.click(
            on_clear_selection,
            inputs=[highlighted_class, trajectory_coords, current_model],
            outputs=[
                selected_image,
                selected_details,
                selected_idx,
                manual_neighbors,
                knn_neighbors,
                knn_distances,
                umap_plot,
                neighbor_gallery,
                neighbor_info,
            ],
        )

        class_dropdown.change(
            on_class_filter,
            inputs=[class_dropdown, selected_idx, manual_neighbors, knn_neighbors, trajectory_coords, current_model],
            outputs=[umap_plot, highlighted_class, class_status],
        )

        clear_class_btn.click(
            on_clear_class,
            inputs=[selected_idx, manual_neighbors, knn_neighbors, trajectory_coords, current_model],
            outputs=[umap_plot, highlighted_class, class_dropdown, class_status],
        )

        if len(visualizer.model_configs) > 1:
            model_dropdown.change(
                on_model_switch,
                inputs=[
                    model_dropdown, current_model, selected_idx, manual_neighbors,
                    knn_neighbors, knn_distances, highlighted_class
                ],
                outputs=[
                    current_model,
                    umap_plot,
                    status_text,
                    model_status,
                    selected_idx,
                    manual_neighbors,
                    knn_neighbors,
                    knn_distances,
                    highlighted_class,
                    trajectory_coords,
                    preview_image,
                    preview_details,
                    selected_image,
                    selected_details,
                    class_dropdown,
                    generated_image,
                    intermediate_gallery,
                    gen_status,
                    intermediate_images,
                    animation_frame,
                    generation_info,
                    neighbor_gallery,
                    neighbor_info,
                ],
            )

        # Note: neighbor display is updated directly in click/suggest handlers
        # No need for state.change() listeners which can cause duplicate events

        # --- Suggest neighbors button ---
        def on_suggest_neighbors(sel_idx, k_val, high_class, man_n, traj, model_name):
            """Auto-suggest K nearest neighbors (preserves trajectory)."""
            if sel_idx is None:
                return gr.update(), [], {}, [], "Select a point first"

            # Clamp k to max 20
            k_val = int(k_val)
            clamped = k_val > 20
            k_val = min(k_val, 20)

            # Find KNN neighbors
            neighbors = visualizer.find_knn_neighbors(model_name, sel_idx, k=k_val)
            if not neighbors:
                return gr.update(), [], {}, [], "No neighbors found"

            # Extract indices and distances
            knn_idx = [idx for idx, _ in neighbors]
            knn_dist = dict(neighbors)

            # Update plot (preserve trajectory)
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_idx,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )

            # Build neighbor gallery
            gallery, info = build_neighbor_gallery(model_name, sel_idx, man_n or [], knn_idx, knn_dist)

            # Add clamped notice if needed
            if clamped:
                info += " (max 20)"

            return fig, knn_idx, knn_dist, gallery, info

        suggest_btn.click(
            on_suggest_neighbors,
            inputs=[selected_idx, knn_k_slider, highlighted_class, manual_neighbors, trajectory_coords, current_model],
            outputs=[umap_plot, knn_neighbors, knn_distances, neighbor_gallery, neighbor_info],
        )

        # --- Clear neighbors button ---
        def on_clear_neighbors(sel_idx, high_class, traj, model_name):
            """Clear all neighbors (preserves trajectory)."""
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )
            return fig, [], [], {}, [], "No neighbors selected"

        clear_neighbors_btn.click(
            on_clear_neighbors,
            inputs=[selected_idx, highlighted_class, trajectory_coords, current_model],
            outputs=[
                umap_plot, manual_neighbors, knn_neighbors,
                knn_distances, neighbor_gallery, neighbor_info
            ],
        )

        # --- Generate button ---
        def on_generate(
            sel_idx, man_n, knn_n, n_steps, m_steps, guidance, s_max, s_min, high_class, existing_traj, model_name
        ):
            """Generate image from selected neighbors with trajectory visualization."""
            existing_traj = existing_traj or []

            # Get model data
            model_data = visualizer.get_model(model_name)
            if model_data is None:
                return None, gr.update(), "Model not loaded", gr.update(), existing_traj, [], None

            # Combine all neighbors
            all_neighbors = list(set((man_n or []) + (knn_n or [])))
            if sel_idx is not None and sel_idx not in all_neighbors:
                all_neighbors.insert(0, sel_idx)

            if not all_neighbors:
                return None, gr.update(), "Select neighbors first", gr.update(), existing_traj, [], None

            # Get class label from selected point (or first neighbor)
            ref_idx = sel_idx if sel_idx is not None else all_neighbors[0]
            if "class_label" in model_data.df.columns:
                class_label = int(model_data.df.iloc[ref_idx]["class_label"])
            else:
                class_label = None

            # Get layers for trajectory extraction
            extract_layers = sorted(model_data.umap_params.get("layers", []))
            can_project = model_data.umap_reducer is not None and len(extract_layers) > 0

            # Load adapter (lazy)
            with visualizer._generation_lock:
                adapter = visualizer.load_adapter(model_name)
                if adapter is None:
                    return None, gr.update(), "Checkpoint not found", gr.update(), [], [], None

                # Prepare activation dict
                activation_dict = visualizer.prepare_activation_dict(model_name, all_neighbors)
                if activation_dict is None:
                    return None, gr.update(), "Failed to prepare activations", gr.update(), [], [], None

                # Create masker and register hooks
                masker = ActivationMasker(adapter)
                for layer_name, activation in activation_dict.items():
                    masker.set_mask(layer_name, activation)
                masker.register_hooks(list(activation_dict.keys()))

                try:
                    # Generate with trajectory, intermediates, and noised inputs
                    result = generate_with_mask_multistep(
                        adapter,
                        masker,
                        class_label=class_label,
                        num_steps=int(n_steps),
                        mask_steps=int(m_steps),
                        sigma_max=float(s_max),
                        sigma_min=float(s_min),
                        guidance_scale=float(guidance),
                        stochastic=True,
                        num_samples=1,
                        device=visualizer.device,
                        extract_layers=extract_layers if can_project else None,
                        return_trajectory=can_project,
                        return_intermediates=True,
                        return_noised_inputs=True,
                    )
                finally:
                    masker.remove_hooks()

            # Unpack results: (images, labels, [trajectory], [intermediates], [noised_inputs])
            images = result[0]
            trajectory_acts = []
            intermediate_imgs = []
            noised_inputs = []
            idx = 2  # Start after images, labels
            if can_project:
                trajectory_acts = result[idx] if len(result) > idx else []
                idx += 1
            intermediate_imgs = result[idx] if len(result) > idx else []
            idx += 1
            noised_inputs = result[idx] if len(result) > idx else []

            # Compute sigma schedule used during this generation (store with results)
            rho = 7.0
            sigmas = []
            for i in range(int(n_steps)):
                ramp = i / max(int(n_steps) - 1, 1)
                min_inv_rho = float(s_min) ** (1 / rho)
                max_inv_rho = float(s_max) ** (1 / rho)
                sigma = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
                sigmas.append(sigma)

            # Project trajectory through UMAP
            traj_coords = []
            if trajectory_acts and model_data.umap_reducer:
                for i, act in enumerate(trajectory_acts):
                    try:
                        # Scale if scaler exists
                        if model_data.umap_scaler is not None:
                            act = model_data.umap_scaler.transform(act)
                        # Project to 2D
                        coords = model_data.umap_reducer.transform(act)
                        sigma = sigmas[i] if i < len(sigmas) else 0.0
                        traj_coords.append((float(coords[0, 0]), float(coords[0, 1]), sigma))
                    except Exception as e:
                        print(f"[Trajectory] Failed to project step {i}: {e}")

            # Append new trajectory to existing list
            all_trajectories = list(existing_traj)
            if traj_coords:
                all_trajectories.append(traj_coords)

            # Build updated plot with all trajectories
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_n or [],
                highlighted_class=high_class,
                trajectory=all_trajectories if all_trajectories else None,
            )

            # Convert to numpy for gr.Image
            gen_img_raw = images[0].numpy()
            class_name = visualizer.get_class_name(class_label) if class_label else "random"
            n_traj = len(all_trajectories) if all_trajectories else 0
            n_steps = len(intermediate_imgs)

            # Create composite for final image (output + last noised input as inset)
            if noised_inputs and len(noised_inputs) > 0:
                last_noised = noised_inputs[-1][0].numpy()
                gen_img = GradioVisualizer.create_composite_image(gen_img_raw, last_noised)
            else:
                gen_img = gen_img_raw

            # Build generation info for frame display
            gen_info = {
                "class_id": class_label,
                "class_name": class_name,
                "n_traj": n_traj,
                "n_steps": n_steps,
            }

            # Status shows final result info
            traj_info = f" | {n_traj} traj" if n_traj else ""
            status = f"Class {class_label}: {class_name}{traj_info} | Final"

            # Build intermediate gallery and state: list of (image, sigma) tuples
            # Each step shows denoised output with noised input as inset
            step_gallery = []
            intermediates_state = []  # For trajectory hover
            for i, step_img in enumerate(intermediate_imgs):
                sigma = sigmas[i] if i < len(sigmas) else 0.0
                img_np = step_img[0].numpy()

                # Create composite with noised input inset if available
                if noised_inputs and i < len(noised_inputs):
                    noised_np = noised_inputs[i][0].numpy()
                    composite_img = GradioVisualizer.create_composite_image(img_np, noised_np)
                else:
                    composite_img = img_np

                caption = f"{class_label}: {class_name} | Step {i+1}/{n_steps} | σ={sigma:.1f}"
                step_gallery.append((composite_img, caption))
                intermediates_state.append((composite_img, sigma))

            # Gallery label (shown when nothing selected)
            gallery_label = f"{class_label}: {class_name} | {n_steps} steps"
            gallery_update = gr.update(value=step_gallery, label=gallery_label)

            return gen_img, gallery_update, status, fig, all_trajectories, intermediates_state, gen_info

        generate_btn.click(
            on_generate,
            inputs=[
                selected_idx, manual_neighbors, knn_neighbors,
                num_steps_slider, mask_steps_slider, guidance_slider,
                sigma_max_input, sigma_min_input, highlighted_class, trajectory_coords, current_model,
            ],
            outputs=[
                generated_image, intermediate_gallery, gen_status,
                umap_plot, trajectory_coords, intermediate_images,
                generation_info
            ],
        )

        # --- Clear generated button ---
        def on_clear_generated(sel_idx, man_n, knn_n, high_class, model_name):
            """Clear generated image, intermediates, and trajectory."""
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_n or [],
                highlighted_class=high_class,
            )
            gallery_update = gr.update(value=[], label="Denoising Steps")
            return None, gallery_update, "Select neighbors, then generate", fig, [], [], -1, None

        clear_gen_btn.click(
            on_clear_generated,
            inputs=[selected_idx, manual_neighbors, knn_neighbors, highlighted_class, current_model],
            outputs=[
                generated_image, intermediate_gallery, gen_status,
                umap_plot, trajectory_coords, intermediate_images,
                animation_frame, generation_info
            ],
        )

        # --- Frame navigation for intermediate images ---
        def format_frame_info(gen_info, frame_idx, n_frames, sigma):
            """Format frame info string with class, step, sigma (compact)."""
            if not gen_info:
                return f"Step {frame_idx + 1}/{n_frames} | σ={sigma:.1f}"

            class_id = gen_info.get("class_id", "?")
            class_name = gen_info.get("class_name", "")

            return f"{class_id}: {class_name} | Step {frame_idx + 1}/{n_frames} | σ={sigma:.1f}"

        def on_next_frame(intermediates, current_frame, gen_info):
            """Show next intermediate frame."""
            if not intermediates:
                return gr.update(), -1, gr.update()

            n_frames = len(intermediates)
            # If at final (-1), go to first frame; otherwise advance
            if current_frame == -1:
                new_frame = 0
            else:
                new_frame = (current_frame + 1) % n_frames

            img, sigma = intermediates[new_frame]
            label = format_frame_info(gen_info, new_frame, n_frames, sigma)
            return img, new_frame, gr.update(label=label)

        def on_prev_frame(intermediates, current_frame, gen_info):
            """Show previous intermediate frame."""
            if not intermediates:
                return gr.update(), -1, gr.update()

            n_frames = len(intermediates)
            # If at final (-1) or first, go to last frame; otherwise go back
            if current_frame <= 0:
                new_frame = n_frames - 1
            else:
                new_frame = current_frame - 1

            img, sigma = intermediates[new_frame]
            label = format_frame_info(gen_info, new_frame, n_frames, sigma)
            return img, new_frame, gr.update(label=label)

        next_frame_btn.click(
            on_next_frame,
            inputs=[intermediate_images, animation_frame, generation_info],
            outputs=[generated_image, animation_frame, intermediate_gallery],
        )

        prev_frame_btn.click(
            on_prev_frame,
            inputs=[intermediate_images, animation_frame, generation_info],
            outputs=[generated_image, animation_frame, intermediate_gallery],
        )

        # Clicking gallery item shows it in main image
        def on_gallery_select(evt: gr.SelectData, intermediates, gen_info):
            """Show selected gallery item in main generated image."""
            if not intermediates or evt.index >= len(intermediates):
                return gr.update(), -1, gr.update()

            img, sigma = intermediates[evt.index]
            n_frames = len(intermediates)
            label = format_frame_info(gen_info, evt.index, n_frames, sigma)
            return img, evt.index, gr.update(label=label)

        intermediate_gallery.select(
            on_gallery_select,
            inputs=[intermediate_images, generation_info],
            outputs=[generated_image, animation_frame, intermediate_gallery],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio Diffusion Activation Visualizer")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to embeddings CSV")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server on")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device (auto-detected if not specified)",
    )
    parser.add_argument("--num-steps", type=int, default=5, help="Number of denoising steps")
    parser.add_argument("--mask-steps", type=int, default=1, help="Steps to apply mask")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--sigma-max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--sigma-min", type=float, default=0.5, help="Minimum sigma")
    parser.add_argument("--max-classes", "-c", type=int, default=None, help="Max classes to load")
    parser.add_argument("--model", "-m", type=str, default=None, help="Initial model to load")
    args = parser.parse_args()

    visualizer = GradioVisualizer(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings,
        device=get_device(args.device),
        num_steps=args.num_steps,
        mask_steps=args.mask_steps,
        guidance_scale=args.guidance_scale,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        max_classes=args.max_classes,
        initial_model=args.model,
    )

    app = create_gradio_app(visualizer)
    app.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        js=PLOTLY_HANDLER_JS,
    )


if __name__ == "__main__":
    main()

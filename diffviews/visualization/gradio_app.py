"""
Gradio-based diffusion activation visualizer.
Port of the Dash visualization app with multi-user support.
"""

import argparse
import json
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class GradioVisualizer:
    """Gradio-based visualizer with multi-user support."""

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
        self.embeddings_path = embeddings_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_steps = num_steps
        self.mask_steps = mask_steps
        self.guidance_scale = guidance_scale
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.label_dropout = label_dropout
        self.adapter_name = adapter_name
        self.max_classes = max_classes

        # Shared state (read-only after init for thread safety)
        self.model_configs: Dict[str, dict] = {}
        self.current_model: Optional[str] = None
        self.df: pd.DataFrame = pd.DataFrame()
        self.activations: Optional[np.ndarray] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.class_labels: Dict[int, str] = {}
        self.umap_reducer = None
        self.umap_scaler = None
        self.umap_params: Dict = {}
        self.layer_shapes: Dict[str, tuple] = {}
        self.nn_model = None

        # Adapter with thread lock for generation
        self.adapter = None
        self._generation_lock = threading.Lock()

        # Discover available models
        self.discover_models()

        # Set initial model
        if initial_model and initial_model in self.model_configs:
            self.current_model = initial_model
        elif "dmd2" in self.model_configs:
            self.current_model = "dmd2"
        elif self.model_configs:
            self.current_model = list(self.model_configs.keys())[0]

        # Set data_dir to current model's directory
        if self.current_model and self.current_model in self.model_configs:
            config = self.model_configs[self.current_model]
            self.data_dir = config["data_dir"]
            self.adapter_name = config["adapter"]
            if config.get("checkpoint"):
                ckpt = Path(config["checkpoint"])
                if not ckpt.is_absolute():
                    candidate = config["data_dir"] / ckpt
                    if candidate.exists():
                        ckpt = candidate
                self.checkpoint_path = ckpt
            if self.embeddings_path is None and config.get("embeddings_path"):
                self.embeddings_path = config["embeddings_path"]
            self.sigma_max = config.get("sigma_max", self.sigma_max)
            self.sigma_min = config.get("sigma_min", self.sigma_min)
            self.num_steps = config.get("default_steps", self.num_steps)
        else:
            self.data_dir = self.root_data_dir

        # Load class labels and data
        self.load_class_labels()
        print(f"Data directory: {self.data_dir.absolute()}")
        self.load_data()

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

    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model.

        Args:
            model_name: Name of the model to switch to

        Returns:
            True if switch succeeded, False otherwise
        """
        if model_name not in self.model_configs:
            print(f"Warning: Model '{model_name}' not found in configs")
            return False

        config = self.model_configs[model_name]
        print(f"Switching to model: {model_name}")

        # Reset model-specific state
        self.adapter = None
        self.layer_shapes = {}
        self.umap_reducer = None
        self.umap_scaler = None
        self.nn_model = None
        self.activations = None
        self.metadata_df = None

        # Update paths and config
        self.current_model = model_name
        self.data_dir = config["data_dir"]
        self.adapter_name = config["adapter"]
        if config.get("checkpoint"):
            ckpt = Path(config["checkpoint"])
            if not ckpt.is_absolute():
                for base in [config["data_dir"], self.root_data_dir, Path.cwd()]:
                    candidate = base / ckpt
                    if candidate.exists():
                        ckpt = candidate
                        break
            self.checkpoint_path = ckpt
        self.embeddings_path = config["embeddings_path"]

        # Apply model-specific defaults
        self.sigma_max = config.get("sigma_max", 80.0)
        self.sigma_min = config.get("sigma_min", 0.5)
        self.num_steps = config.get("default_steps", 5)

        # Reload data
        self.load_data()
        self.fit_nearest_neighbors()

        return True

    def load_class_labels(self):
        """Load ImageNet class labels."""
        label_path = self.root_data_dir / "imagenet_standard_class_index.json"
        if not label_path.exists():
            label_path = self.data_dir / "imagenet_standard_class_index.json"
        if not label_path.exists():
            label_path = self.data_dir / "imagenet_class_labels.json"
        if not label_path.exists():
            label_path = (
                Path(__file__).parent.parent / "data" / "imagenet_standard_class_index.json"
            )

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
                self.class_labels = {int(k): v[1] for k, v in raw_labels.items()}
            print(f"Loaded {len(self.class_labels)} ImageNet class labels")
        else:
            print(f"Warning: Class labels not found at {label_path}")
            self.class_labels = {}

    def get_class_name(self, class_id: int) -> str:
        """Get human-readable class name for a class ID."""
        if class_id in self.class_labels:
            return self.class_labels[class_id]
        return f"Unknown class {class_id}"

    def load_data(self):
        """Load embeddings and activations."""
        if self.embeddings_path:
            if not Path(self.embeddings_path).exists():
                print(f"ERROR: Embeddings file not found: {self.embeddings_path}")
                self.df = pd.DataFrame()
                return
            print(f"Loading embeddings from {self.embeddings_path}")
            self.df = pd.read_csv(self.embeddings_path)

            # Load UMAP params
            param_path = Path(self.embeddings_path).with_suffix(".json")
            if param_path.exists():
                with open(param_path, "r", encoding="utf-8") as f:
                    self.umap_params = json.load(f)
            else:
                self.umap_params = {}

            # Load UMAP model for inverse_transform (optional)
            model_path = Path(self.embeddings_path).with_suffix(".pkl")
            if model_path.exists():
                print(f"Loading UMAP model from {model_path}")
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                    self.umap_reducer = model_data["reducer"]
                    self.umap_scaler = model_data["scaler"]
                print("UMAP model loaded")

            # Filter by max_classes if specified
            if self.max_classes is not None and "class_label" in self.df.columns:
                unique_classes = self.df["class_label"].unique()
                if len(unique_classes) > self.max_classes:
                    classes_to_keep = unique_classes[: self.max_classes]
                    original_count = len(self.df)
                    self.df = self.df[self.df["class_label"].isin(classes_to_keep)].reset_index(
                        drop=True
                    )
                    print(
                        f"Filtered to {self.max_classes} classes: "
                        f"{original_count} -> {len(self.df)} samples"
                    )

            # Load activations for generation
            if self.activations is None:
                self.activations, self.metadata_df = self.load_activations_for_model(
                    "imagenet_real"
                )

            print(f"Loaded {len(self.df)} samples")
        else:
            print("No embeddings found.")
            self.df = pd.DataFrame()

    def load_activations_for_model(
        self, model_type: str
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Load raw activations for generation."""
        activation_dir = self.data_dir / "activations" / model_type
        metadata_path = self.data_dir / "metadata" / model_type / "dataset_info.json"

        if not activation_dir.exists():
            print(f"Warning: Activation dir not found: {activation_dir}")
            return None, None

        if not metadata_path.exists():
            print(f"Warning: No metadata found: {metadata_path}")
            return None, None

        activations, metadata_df = load_dataset_activations(activation_dir, metadata_path)
        return activations, metadata_df

    def fit_nearest_neighbors(self):
        """Fit KNN model on UMAP coordinates."""
        if self.df.empty or "umap_x" not in self.df.columns:
            return

        umap_coords = self.df[["umap_x", "umap_y"]].values
        print(f"[KNN] Fitting on UMAP coordinates: {umap_coords.shape}")
        self.nn_model = NearestNeighbors(n_neighbors=21, metric="euclidean")
        self.nn_model.fit(umap_coords)

    def find_knn_neighbors(
        self, idx: int, k: int = 5, exclude_selected: bool = True
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a point.

        Args:
            idx: Index of the query point
            k: Number of neighbors to return
            exclude_selected: If True, exclude the query point from results

        Returns:
            List of (neighbor_idx, distance) tuples sorted by distance
        """
        if self.nn_model is None or idx >= len(self.df):
            return []

        # Query k+1 since the point itself is included
        query_k = k + 1 if exclude_selected else k
        query_point = self.df.iloc[idx][["umap_x", "umap_y"]].values.reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(query_point, n_neighbors=query_k)

        results = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            if exclude_selected and neighbor_idx == idx:
                continue
            results.append((int(neighbor_idx), float(dist)))

        return results[:k]

    def get_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image as numpy array for gr.Image."""
        try:
            full_path = self.data_dir / image_path
            return np.array(Image.open(full_path))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_adapter(self):
        """Lazily load the generation adapter."""
        if self.adapter is not None:
            return self.adapter

        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            print(f"Error: checkpoint not found at {self.checkpoint_path}")
            return None

        print(f"Loading adapter: {self.adapter_name} from {self.checkpoint_path}")
        AdapterClass = get_adapter(self.adapter_name)
        self.adapter = AdapterClass.from_checkpoint(
            self.checkpoint_path,
            device=self.device,
            label_dropout=self.label_dropout,
        )
        # Cache layer shapes
        self.layer_shapes = self.adapter.get_layer_shapes()
        print(f"Adapter loaded. Layers: {list(self.layer_shapes.keys())}")
        return self.adapter

    def prepare_activation_dict(
        self, neighbor_indices: List[int]
    ) -> Optional[Dict[str, "torch.Tensor"]]:
        """Prepare activation dict for masked generation.

        Args:
            neighbor_indices: List of indices to average activations from

        Returns:
            Dict mapping layer_name -> (1, C, H, W) tensor, or None if error
        """
        import torch

        if self.activations is None or len(neighbor_indices) == 0:
            return None

        # Get layers from UMAP params (must match order used during embedding)
        layers = sorted(self.umap_params.get("layers", ["encoder_bottleneck", "midblock"]))
        if not self.layer_shapes:
            if self.adapter is None:
                self.load_adapter()
            if self.adapter is None:
                return None

        # Average neighbor activations in high-D space
        neighbor_acts = self.activations[neighbor_indices]  # (N, D)
        center_activation = np.mean(neighbor_acts, axis=0, keepdims=True)  # (1, D)

        # Split into per-layer activations (MUST be sorted order!)
        activation_dict = {}
        offset = 0
        for layer_name in layers:
            if layer_name not in self.layer_shapes:
                print(f"Warning: layer {layer_name} not in adapter shapes")
                continue
            shape = self.layer_shapes[layer_name]  # (C, H, W)
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
        selected_idx: Optional[int] = None,
        manual_neighbors: Optional[List[int]] = None,
        knn_neighbors: Optional[List[int]] = None,
        highlighted_class: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get DataFrame for ScatterPlot with highlight column."""
        if self.df.empty:
            return pd.DataFrame(columns=["umap_x", "umap_y", "highlight", "sample_id"])

        # Create copy with highlight column
        plot_df = self.df[["umap_x", "umap_y", "sample_id"]].copy()
        if "class_label" in self.df.columns:
            plot_df["class_label"] = self.df["class_label"].astype(str)

        # Add highlight column for visual distinction
        plot_df["highlight"] = "normal"

        manual_neighbors = manual_neighbors or []
        knn_neighbors = knn_neighbors or []

        # Mark class highlights
        if highlighted_class is not None and "class_label" in self.df.columns:
            class_mask = self.df["class_label"] == highlighted_class
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

    def get_class_options(self) -> List[Tuple[str, int]]:
        """Get class options for dropdown."""
        if self.df.empty or "class_label" not in self.df.columns:
            return []

        unique_classes = self.df["class_label"].dropna().unique()
        unique_classes = sorted([int(c) for c in unique_classes])

        return [(f"{c}: {self.get_class_name(c)}", c) for c in unique_classes]

    def get_color_map(self) -> dict:
        """Generate color map for class labels using plasma colormap."""
        if self.df.empty or "class_label" not in self.df.columns:
            return {}

        import matplotlib.pyplot as plt
        unique_classes = self.df["class_label"].dropna().unique()
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
        selected_idx: Optional[int] = None,
        manual_neighbors: Optional[List[int]] = None,
        knn_neighbors: Optional[List[int]] = None,
        highlighted_class: Optional[int] = None,
        trajectory: Optional[List[Tuple[float, float, float]]] = None,
    ) -> go.Figure:
        """Create Plotly figure for UMAP scatter plot.

        Args:
            selected_idx: Index of currently selected point
            manual_neighbors: List of manually selected neighbor indices
            knn_neighbors: List of KNN neighbor indices
            highlighted_class: Class ID to highlight
            trajectory: List of (x, y, sigma) tuples for denoising trajectory

        Returns:
            Plotly Figure object
        """
        if self.df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data loaded")
            return fig

        manual_neighbors = manual_neighbors or []
        knn_neighbors = knn_neighbors or []

        # Get color map for classes
        color_map = self.get_color_map()

        # Build customdata with df indices for click handling
        customdata = list(range(len(self.df)))

        # Determine point colors based on class
        if "class_label" in self.df.columns:
            colors = [color_map.get(str(int(c)), "#888888") for c in self.df["class_label"]]
        else:
            colors = ["#1f77b4"] * len(self.df)

        # Create main scatter trace
        fig = go.Figure()

        # Add main data points (use scattergl for better performance)
        fig.add_trace(go.Scattergl(
            x=self.df["umap_x"].tolist(),
            y=self.df["umap_y"].tolist(),
            mode="markers",
            marker=dict(
                size=6,
                color=colors,
                opacity=0.7,
            ),
            customdata=customdata,
            hovertemplate="<b>%{customdata}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            name="samples",
        ))

        # Highlight class if specified
        if highlighted_class is not None and "class_label" in self.df.columns:
            class_mask = self.df["class_label"] == highlighted_class
            class_indices = self.df[class_mask].index.tolist()
            if class_indices:
                fig.add_trace(go.Scattergl(
                    x=self.df.loc[class_mask, "umap_x"].tolist(),
                    y=self.df.loc[class_mask, "umap_y"].tolist(),
                    mode="markers",
                    marker=dict(size=10, color="yellow", line=dict(width=1, color="black")),
                    customdata=[int(i) for i in class_indices],
                    hoverinfo="skip",
                    name="class_highlight",
                    showlegend=False,
                ))

        # Highlight KNN neighbors (green ring)
        if knn_neighbors:
            knn_df = self.df.iloc[knn_neighbors]
            fig.add_trace(go.Scattergl(
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
            man_df = self.df.iloc[manual_neighbors]
            fig.add_trace(go.Scattergl(
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
        if selected_idx is not None and selected_idx < len(self.df):
            sel_row = self.df.iloc[selected_idx]
            fig.add_trace(go.Scattergl(
                x=[float(sel_row["umap_x"])],
                y=[float(sel_row["umap_y"])],
                mode="markers",
                marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color="red")),
                customdata=[selected_idx],
                hoverinfo="skip",
                name="selected",
                showlegend=False,
            ))

        # Denoising trajectory (if available)
        if trajectory and len(trajectory) > 1:
            traj_x = [t[0] for t in trajectory]
            traj_y = [t[1] for t in trajectory]
            traj_sigma = [t[2] for t in trajectory]

            # Line trace for trajectory path
            fig.add_trace(go.Scatter(
                x=traj_x,
                y=traj_y,
                mode="lines",
                line=dict(color="lime", width=3),
                hoverinfo="skip",
                name="trajectory_line",
                showlegend=False,
            ))

            # Markers with sigma labels
            fig.add_trace(go.Scatter(
                x=traj_x,
                y=traj_y,
                mode="markers+text",
                marker=dict(
                    size=12,
                    color=list(range(len(trajectory))),
                    colorscale="Viridis",
                    line=dict(width=1, color="white"),
                ),
                text=[f"σ={s:.1f}" for s in traj_sigma],
                textposition="top center",
                textfont=dict(size=9, color="white"),
                hovertemplate="Step %{customdata}<br>σ=%{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
                customdata=list(range(len(trajectory))),
                name="trajectory",
                showlegend=False,
            ))

            # Start marker (star)
            fig.add_trace(go.Scatter(
                x=[traj_x[0]],
                y=[traj_y[0]],
                mode="markers",
                marker=dict(symbol="star", size=18, color="lime", line=dict(width=1, color="white")),
                hovertemplate="Start (σ=%.1f)<extra></extra>" % traj_sigma[0],
                name="traj_start",
                showlegend=False,
            ))

            # End marker (diamond)
            fig.add_trace(go.Scatter(
                x=[traj_x[-1]],
                y=[traj_y[-1]],
                mode="markers",
                marker=dict(symbol="diamond", size=14, color="red", line=dict(width=1, color="white")),
                hovertemplate="End (σ=%.1f)<extra></extra>" % traj_sigma[-1],
                name="traj_end",
                showlegend=False,
            ))

        # Layout
        fig.update_layout(
            title="Activation UMAP",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode="closest",
            template="plotly_white",
            showlegend=False,
            height=550,
            margin=dict(l=50, r=20, t=40, b=50),
        )

        return fig


# JavaScript for Plotly click handling - sends click data to hidden textbox
CLICK_HANDLER_JS = """
function attachPlotlyClickHandler() {
    console.log('[PlotlyClick] Attempting to attach handler...');

    // Find Plotly plot - try multiple selectors
    let plotDiv = document.querySelector('#umap-plot .plotly-graph-div');
    if (!plotDiv) plotDiv = document.querySelector('#umap-plot .js-plotly-plot');
    if (!plotDiv) plotDiv = document.querySelector('#umap-plot [class*="plotly"]');

    // Find textbox - Gradio may use input or textarea
    let clickBox = document.querySelector('#click-data-box textarea');
    if (!clickBox) clickBox = document.querySelector('#click-data-box input');

    console.log('[PlotlyClick] plotDiv:', plotDiv);
    console.log('[PlotlyClick] clickBox:', clickBox);

    if (!plotDiv || !clickBox) {
        console.log('[PlotlyClick] Elements not found, retrying in 200ms...');
        setTimeout(attachPlotlyClickHandler, 200);
        return;
    }

    // Avoid duplicate handlers
    if (plotDiv._gradioClickHandlerAttached) {
        console.log('[PlotlyClick] Handler already attached');
        return;
    }
    plotDiv._gradioClickHandlerAttached = true;

    plotDiv.on('plotly_click', function(data) {
        console.log('[PlotlyClick] Click detected:', data);
        if (!data || !data.points || data.points.length === 0) return;

        const point = data.points[0];
        const clickData = {
            pointIndex: point.customdata,
            x: point.x,
            y: point.y,
            curveNumber: point.curveNumber
        };

        console.log('[PlotlyClick] Sending:', clickData);

        // Set value and trigger input event for Gradio
        clickBox.value = JSON.stringify(clickData);
        clickBox.dispatchEvent(new Event('input', { bubbles: true }));
    });

    console.log('[PlotlyClick] Handler attached successfully!');
}

// Attach handler after plot renders
setTimeout(attachPlotlyClickHandler, 500);
"""


def create_gradio_app(visualizer: GradioVisualizer) -> gr.Blocks:
    """Create Gradio Blocks app."""

    # CSS for plot and hidden elements
    custom_css = """
    #umap-plot {
        min-height: 550px !important;
    }
    #click-data-box {
        display: none !important;
    }
    """

    with gr.Blocks(
        title="Diffusion Activation Visualizer",
        theme=gr.themes.Soft(),
        css=custom_css,
        head=f"<script>{CLICK_HANDLER_JS}</script>",
    ) as app:

        # Per-session state
        selected_idx = gr.State(value=None)
        manual_neighbors = gr.State(value=[])
        knn_neighbors = gr.State(value=[])
        knn_distances = gr.State(value={})  # {idx: distance} for KNN neighbors
        highlighted_class = gr.State(value=None)
        trajectory_coords = gr.State(value=[])  # [(x, y, sigma), ...] for denoising path

        gr.Markdown("# Diffusion Activation Visualizer")

        with gr.Row():
            # Left column (sidebar)
            with gr.Column(scale=1):
                # Model selector (only if multiple models)
                if len(visualizer.model_configs) > 1:
                    model_dropdown = gr.Dropdown(
                        choices=list(visualizer.model_configs.keys()),
                        value=visualizer.current_model,
                        label="Model",
                        interactive=True,
                    )
                    model_status = gr.Markdown("")
                else:
                    model_dropdown = gr.Dropdown(
                        choices=[visualizer.current_model] if visualizer.current_model else [],
                        value=visualizer.current_model,
                        label="Model",
                        visible=False,
                    )
                    model_status = gr.Markdown(visible=False)

                # Preview section
                with gr.Group():
                    gr.Markdown("### Preview")
                    preview_image = gr.Image(label=None, show_label=False, height=200)
                    preview_details = gr.Markdown("Click a point to preview")

                # Class filter
                with gr.Group():
                    gr.Markdown("### Class Filter")
                    class_dropdown = gr.Dropdown(
                        choices=visualizer.get_class_options(),
                        label="Select class",
                        interactive=True,
                    )
                    clear_class_btn = gr.Button("Clear Highlight", size="sm")
                    class_status = gr.Markdown("")

            # Center column (main plot)
            with gr.Column(scale=3, min_width=600):
                # Hidden textbox for JS bridge (receives click data from Plotly)
                click_data_box = gr.Textbox(
                    value="",
                    elem_id="click-data-box",
                    visible=False,
                )
                # Use Plotly via gr.Plot for proper click handling
                umap_plot = gr.Plot(
                    value=visualizer.create_umap_figure(),
                    elem_id="umap-plot",
                    show_label=False,
                )
                status_text = gr.Markdown(
                    f"Showing {len(visualizer.df)} samples"
                    + (f" ({visualizer.current_model})" if visualizer.current_model else "")
                )

            # Right column (selection & controls)
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Selected Sample")
                    selected_image = gr.Image(label=None, show_label=False, height=200)
                    selected_details = gr.Markdown("Click a point to select")
                    clear_selection_btn = gr.Button("Clear Selection", size="sm")

                # Generation settings
                with gr.Group():
                    gr.Markdown("### Generation")
                    generated_image = gr.Image(label=None, show_label=False, height=200)
                    with gr.Row():
                        num_steps_slider = gr.Slider(
                            1, 50, value=visualizer.num_steps, step=1, label="Steps"
                        )
                        mask_steps_slider = gr.Slider(
                            1, 50, value=visualizer.mask_steps, step=1, label="Mask Steps"
                        )
                    guidance_slider = gr.Slider(
                        -10, 20, value=visualizer.guidance_scale, step=0.1, label="Guidance"
                    )
                    with gr.Row():
                        sigma_max_input = gr.Number(value=visualizer.sigma_max, label="σ max")
                        sigma_min_input = gr.Number(value=visualizer.sigma_min, label="σ min")

                    generate_btn = gr.Button("Generate from Neighbors", variant="primary")
                    gen_status = gr.Markdown("Select neighbors, then generate")

                # Neighbor list
                with gr.Group():
                    gr.Markdown("### Neighbors")
                    with gr.Row():
                        knn_k_slider = gr.Slider(
                            1, 10, value=5, step=1, label="K neighbors"
                        )
                        suggest_btn = gr.Button("Suggest", size="sm")
                    neighbor_gallery = gr.Gallery(
                        label=None,
                        show_label=False,
                        columns=2,
                        height=200,
                        object_fit="contain",
                    )
                    neighbor_info = gr.Markdown("No neighbors selected")
                    clear_neighbors_btn = gr.Button("Clear Neighbors", size="sm")

        # --- Event Handlers ---

        def on_load():
            """Initialize KNN model on load."""
            if visualizer.nn_model is None and not visualizer.df.empty:
                visualizer.fit_nearest_neighbors()

        def build_neighbor_gallery(sel_idx, man_n, knn_n, knn_dist):
            """Build neighbor gallery and info text."""
            if sel_idx is None:
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
            for idx in all_neighbors[:10]:
                if idx < len(visualizer.df):
                    sample = visualizer.df.iloc[idx]
                    img = visualizer.get_image(sample["image_path"])
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

        def on_click_data(click_json, sel_idx, man_n, knn_n, knn_dist, high_class, traj):
            """Handle plot click via JS bridge - select point or toggle neighbor."""
            if not click_json:
                return (gr.update(),) * 11

            try:
                click_data = json.loads(click_json)
                point_idx = click_data.get("pointIndex")
                if point_idx is None:
                    return (gr.update(),) * 11
                point_idx = int(point_idx)
            except (json.JSONDecodeError, TypeError, ValueError):
                return (gr.update(),) * 11

            if point_idx < 0 or point_idx >= len(visualizer.df):
                return (gr.update(),) * 11

            knn_dist = knn_dist or {}

            # First click: select this point (clears trajectory)
            if sel_idx is None:
                sample = visualizer.df.iloc[point_idx]
                img = visualizer.get_image(sample["image_path"])

                # Format details
                if "class_label" in sample:
                    class_name = visualizer.get_class_name(int(sample["class_label"]))
                else:
                    class_name = "N/A"
                details = f"**{sample['sample_id']}**\n\n"
                if "class_label" in sample:
                    details += f"Class: {int(sample['class_label'])}: {class_name}\n\n"
                details += f"Coords: ({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"

                # Build updated Plotly figure with selection (no trajectory)
                fig = visualizer.create_umap_figure(
                    selected_idx=point_idx,
                    highlighted_class=high_class
                )

                return (
                    img,           # preview_image
                    details,       # preview_details
                    img,           # selected_image
                    details,       # selected_details
                    point_idx,     # selected_idx
                    [],            # manual_neighbors
                    [],            # knn_neighbors
                    fig,           # umap_plot
                    [],            # neighbor_gallery
                    "Click points or use Suggest",  # neighbor_info
                    [],            # trajectory_coords (cleared)
                )

            # Clicking same point: do nothing
            if point_idx == sel_idx:
                return (gr.update(),) * 11

            # Toggle neighbor (preserve trajectory)
            man_n = list(man_n) if man_n else []
            knn_n = list(knn_n) if knn_n else []

            if point_idx in man_n:
                man_n.remove(point_idx)
            elif point_idx in knn_n:
                knn_n.remove(point_idx)
            else:
                man_n.append(point_idx)

            # Rebuild Plotly figure with updated highlights (preserve trajectory)
            fig = visualizer.create_umap_figure(
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )

            # Build gallery for neighbors
            gallery, info = build_neighbor_gallery(sel_idx, man_n, knn_n, knn_dist)

            return (
                gr.update(),   # preview_image
                gr.update(),   # preview_details
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

        def on_clear_selection(high_class):
            """Clear selection, neighbors, and trajectory."""
            fig = visualizer.create_umap_figure(highlighted_class=high_class)
            return (
                None,                      # preview_image
                "Click a point to preview", # preview_details
                None,                      # selected_image
                "Click a point to select", # selected_details
                None,                      # selected_idx
                [],                        # manual_neighbors
                [],                        # knn_neighbors
                {},                        # knn_distances
                fig,                       # umap_plot
                [],                        # neighbor_gallery
                "No neighbors selected",   # neighbor_info
                [],                        # trajectory_coords
            )

        def on_class_filter(class_value, sel_idx, man_n, knn_n, traj):
            """Handle class filter selection (preserves trajectory)."""
            fig = visualizer.create_umap_figure(
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=class_value,
                trajectory=traj if traj else None,
            )

            if class_value is not None and "class_label" in visualizer.df.columns:
                count = (visualizer.df["class_label"] == class_value).sum()
                status = f"{count} samples"
            else:
                status = ""

            return fig, class_value, status

        def on_clear_class(sel_idx, man_n, knn_n, traj):
            """Clear class highlight (preserves trajectory)."""
            fig = visualizer.create_umap_figure(
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                trajectory=traj if traj else None,
            )
            return fig, None, None, ""

        def on_model_switch(model_name, _sel_idx, _man_n, _knn_n, _knn_dist, _high_class):
            """Handle model switching (resets all state, inputs unused)."""
            if model_name == visualizer.current_model:
                return (gr.update(),) * 14

            success = visualizer.switch_model(model_name)
            if not success:
                return (gr.update(),) * 14

            fig = visualizer.create_umap_figure()
            status = f"Showing {len(visualizer.df)} samples ({model_name})"

            return (
                fig,                               # umap_plot
                status,                            # status_text
                f"Switched to {model_name}",       # model_status
                None,                              # selected_idx
                [],                                # manual_neighbors
                [],                                # knn_neighbors
                {},                                # knn_distances
                None,                              # highlighted_class
                [],                                # trajectory_coords
                None,                              # preview_image
                "Click a point to preview",        # preview_details
                None,                              # selected_image
                "Click a point to select",         # selected_details
                visualizer.get_class_options(),    # class_dropdown choices
            )

        # Wire up events
        # on_load initializes KNN model
        app.load(on_load, outputs=[])

        # Click handling via JS bridge (click_data_box receives JSON from Plotly click)
        click_data_box.input(
            on_click_data,
            inputs=[
                click_data_box, selected_idx, manual_neighbors,
                knn_neighbors, knn_distances, highlighted_class, trajectory_coords
            ],
            outputs=[
                preview_image,
                preview_details,
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
            inputs=[highlighted_class],
            outputs=[
                preview_image,
                preview_details,
                selected_image,
                selected_details,
                selected_idx,
                manual_neighbors,
                knn_neighbors,
                knn_distances,
                umap_plot,
                neighbor_gallery,
                neighbor_info,
                trajectory_coords,
            ],
        )

        class_dropdown.change(
            on_class_filter,
            inputs=[class_dropdown, selected_idx, manual_neighbors, knn_neighbors, trajectory_coords],
            outputs=[umap_plot, highlighted_class, class_status],
        )

        clear_class_btn.click(
            on_clear_class,
            inputs=[selected_idx, manual_neighbors, knn_neighbors, trajectory_coords],
            outputs=[umap_plot, highlighted_class, class_dropdown, class_status],
        )

        if len(visualizer.model_configs) > 1:
            model_dropdown.change(
                on_model_switch,
                inputs=[
                    model_dropdown, selected_idx, manual_neighbors,
                    knn_neighbors, knn_distances, highlighted_class
                ],
                outputs=[
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
                ],
            )

        # Note: neighbor display is updated directly in click/suggest handlers
        # No need for state.change() listeners which can cause duplicate events

        # --- Suggest neighbors button ---
        def on_suggest_neighbors(sel_idx, k_val, high_class, man_n, traj):
            """Auto-suggest K nearest neighbors (preserves trajectory)."""
            if sel_idx is None:
                return gr.update(), [], {}, gr.update(), "Select a point first"

            # Find KNN neighbors
            neighbors = visualizer.find_knn_neighbors(sel_idx, k=int(k_val))
            if not neighbors:
                return gr.update(), [], {}, gr.update(), "No neighbors found"

            # Extract indices and distances
            knn_idx = [idx for idx, _ in neighbors]
            knn_dist = dict(neighbors)

            # Update plot (preserve trajectory)
            fig = visualizer.create_umap_figure(
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_idx,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )

            return fig, knn_idx, knn_dist, gr.update(), f"Found {len(knn_idx)} neighbors"

        suggest_btn.click(
            on_suggest_neighbors,
            inputs=[selected_idx, knn_k_slider, highlighted_class, manual_neighbors, trajectory_coords],
            outputs=[umap_plot, knn_neighbors, knn_distances, neighbor_gallery, neighbor_info],
        )

        # --- Clear neighbors button ---
        def on_clear_neighbors(sel_idx, high_class, traj):
            """Clear all neighbors (preserves trajectory)."""
            fig = visualizer.create_umap_figure(
                selected_idx=sel_idx,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )
            return fig, [], [], {}, [], "No neighbors selected"

        clear_neighbors_btn.click(
            on_clear_neighbors,
            inputs=[selected_idx, highlighted_class, trajectory_coords],
            outputs=[
                umap_plot, manual_neighbors, knn_neighbors,
                knn_distances, neighbor_gallery, neighbor_info
            ],
        )

        # --- Generate button ---
        def on_generate(
            sel_idx, man_n, knn_n, n_steps, m_steps, guidance, s_max, s_min, high_class
        ):
            """Generate image from selected neighbors with trajectory visualization."""
            # Combine all neighbors
            all_neighbors = list(set((man_n or []) + (knn_n or [])))
            if sel_idx is not None and sel_idx not in all_neighbors:
                all_neighbors.insert(0, sel_idx)

            if not all_neighbors:
                return None, "Select neighbors first", gr.update(), []

            # Get class label from selected point (or first neighbor)
            ref_idx = sel_idx if sel_idx is not None else all_neighbors[0]
            if "class_label" in visualizer.df.columns:
                class_label = int(visualizer.df.iloc[ref_idx]["class_label"])
            else:
                class_label = None

            # Get layers for trajectory extraction
            extract_layers = sorted(visualizer.umap_params.get("layers", []))
            can_project = visualizer.umap_reducer is not None and len(extract_layers) > 0

            # Load adapter (lazy)
            with visualizer._generation_lock:
                adapter = visualizer.load_adapter()
                if adapter is None:
                    return None, "Checkpoint not found", gr.update(), []

                # Prepare activation dict
                activation_dict = visualizer.prepare_activation_dict(all_neighbors)
                if activation_dict is None:
                    return None, "Failed to prepare activations", gr.update(), []

                # Create masker and register hooks
                masker = ActivationMasker(adapter)
                for layer_name, activation in activation_dict.items():
                    masker.set_mask(layer_name, activation)
                masker.register_hooks(list(activation_dict.keys()))

                try:
                    # Generate with trajectory extraction
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
                    )
                finally:
                    masker.remove_hooks()

            # Unpack results
            if can_project and len(result) >= 3:
                images, _labels, trajectory_acts = result[:3]
            else:
                images = result[0]
                trajectory_acts = []

            # Project trajectory through UMAP
            traj_coords = []
            if trajectory_acts and visualizer.umap_reducer:
                # Compute sigma schedule (same as generator)
                rho = 7.0
                sigmas = []
                for i in range(int(n_steps)):
                    ramp = i / max(int(n_steps) - 1, 1)
                    min_inv_rho = float(s_min) ** (1 / rho)
                    max_inv_rho = float(s_max) ** (1 / rho)
                    sigma = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
                    sigmas.append(sigma)

                for i, act in enumerate(trajectory_acts):
                    try:
                        # Scale if scaler exists
                        if visualizer.umap_scaler is not None:
                            act = visualizer.umap_scaler.transform(act)
                        # Project to 2D
                        coords = visualizer.umap_reducer.transform(act)
                        sigma = sigmas[i] if i < len(sigmas) else 0.0
                        traj_coords.append((float(coords[0, 0]), float(coords[0, 1]), sigma))
                    except Exception as e:
                        print(f"[Trajectory] Failed to project step {i}: {e}")

            # Build updated plot with trajectory
            fig = visualizer.create_umap_figure(
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_n or [],
                highlighted_class=high_class,
                trajectory=traj_coords if traj_coords else None,
            )

            # Convert to numpy for gr.Image
            gen_img = images[0].numpy()
            class_name = visualizer.get_class_name(class_label) if class_label else "random"
            traj_info = f", {len(traj_coords)} trajectory steps" if traj_coords else ""
            status = f"Generated (class {class_label}: {class_name}{traj_info})"

            return gen_img, status, fig, traj_coords

        generate_btn.click(
            on_generate,
            inputs=[
                selected_idx, manual_neighbors, knn_neighbors,
                num_steps_slider, mask_steps_slider, guidance_slider,
                sigma_max_input, sigma_min_input, highlighted_class,
            ],
            outputs=[generated_image, gen_status, umap_plot, trajectory_coords],
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
    )


if __name__ == "__main__":
    main()

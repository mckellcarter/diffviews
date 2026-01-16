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

# Plotly imports kept for future phases (trajectory visualization)
# import plotly.express as px
# import plotly.graph_objects as go

from diffviews.processing.umap import load_dataset_activations
from diffviews.utils.device import get_device


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
        """Generate color map for class labels using turbo colormap."""
        if self.df.empty or "class_label" not in self.df.columns:
            return {}

        import matplotlib.pyplot as plt
        unique_classes = self.df["class_label"].dropna().unique()
        unique_classes = sorted([int(c) for c in unique_classes])

        if not unique_classes:
            return {}

        cmap = plt.cm.turbo
        color_map = {}
        for i, cls in enumerate(unique_classes):
            rgba = cmap(i / max(len(unique_classes) - 1, 1))
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            color_map[str(cls)] = hex_color
        return color_map


def create_gradio_app(visualizer: GradioVisualizer) -> gr.Blocks:
    """Create Gradio Blocks app."""

    # CSS to make plot larger
    custom_css = """
    #umap-plot {
        min-height: 550px !important;
    }
    #umap-plot .vega-embed {
        width: 100% !important;
    }
    #umap-plot canvas {
        max-width: 100% !important;
    }
    """

    with gr.Blocks(
        title="Diffusion Activation Visualizer",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        # Per-session state
        selected_idx = gr.State(value=None)
        manual_neighbors = gr.State(value=[])
        knn_neighbors = gr.State(value=[])
        knn_distances = gr.State(value={})  # {idx: distance} for KNN neighbors
        highlighted_class = gr.State(value=None)

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
                # Use ScatterPlot for native selection support
                umap_plot = gr.ScatterPlot(
                    value=visualizer.get_plot_dataframe(),
                    x="umap_x",
                    y="umap_y",
                    color="class_label" if "class_label" in visualizer.df.columns else None,
                    title="Activation UMAP",
                    x_title="UMAP 1",
                    y_title="UMAP 2",
                    height=550,
                    color_map=visualizer.get_color_map() or None,
                    elem_id="umap-plot",
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

                # Generation settings (placeholders for phase 2)
                with gr.Group():
                    gr.Markdown("### Generation Settings")
                    with gr.Row():
                        _num_steps_slider = gr.Slider(
                            1, 50, value=visualizer.num_steps, step=1, label="Steps"
                        )
                        _mask_steps_slider = gr.Slider(
                            1, 50, value=visualizer.mask_steps, step=1, label="Mask Steps"
                        )
                    _guidance_slider = gr.Slider(
                        -10, 20, value=visualizer.guidance_scale, step=0.1, label="Guidance"
                    )
                    with gr.Row():
                        _sigma_max_input = gr.Number(value=visualizer.sigma_max, label="σ max")
                        _sigma_min_input = gr.Number(value=visualizer.sigma_min, label="σ min")

                    _generate_btn = gr.Button(
                        "Generate Image", variant="primary", interactive=False
                    )
                    _gen_status = gr.Markdown("*Generation available in phase 2*")

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

        def on_plot_select(
            evt: gr.SelectData, sel_idx, man_n, knn_n, high_class
        ):
            """Handle plot click - select point or toggle neighbor."""
            print(f"[SELECT] evt={evt}, evt.index={getattr(evt, 'index', None)}, "
                  f"evt.value={getattr(evt, 'value', None)}")
            if evt is None or evt.index is None:
                print("[SELECT] No event or index, returning no-op")
                return (gr.update(),) * 8

            # Get point index from selection event
            # ScatterPlot select returns row index
            point_idx = evt.index
            print(f"[SELECT] point_idx={point_idx}, type={type(point_idx)}")
            if isinstance(point_idx, (list, tuple)):
                point_idx = point_idx[0]

            # First click: select this point
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

                # Build updated plot dataframe with selection
                plot_df = visualizer.get_plot_dataframe(
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
                    plot_df,       # umap_plot
                )

            # Clicking same point: do nothing
            if point_idx == sel_idx:
                return (gr.update(),) * 8

            # Toggle neighbor
            man_n = list(man_n) if man_n else []
            knn_n = list(knn_n) if knn_n else []

            if point_idx in man_n:
                man_n.remove(point_idx)
            elif point_idx in knn_n:
                knn_n.remove(point_idx)
            else:
                man_n.append(point_idx)

            # Rebuild plot dataframe with updated highlights
            plot_df = visualizer.get_plot_dataframe(
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=high_class
            )

            return (
                gr.update(),   # preview_image
                gr.update(),   # preview_details
                gr.update(),   # selected_image
                gr.update(),   # selected_details
                sel_idx,       # selected_idx (unchanged)
                man_n,         # manual_neighbors
                knn_n,         # knn_neighbors
                plot_df,       # umap_plot
            )

        def on_clear_selection(high_class):
            """Clear selection and neighbors."""
            plot_df = visualizer.get_plot_dataframe(highlighted_class=high_class)
            return (
                None,                      # preview_image
                "Click a point to preview", # preview_details
                None,                      # selected_image
                "Click a point to select", # selected_details
                None,                      # selected_idx
                [],                        # manual_neighbors
                [],                        # knn_neighbors
                {},                        # knn_distances
                plot_df,                   # umap_plot
                [],                        # neighbor_gallery
                "No neighbors selected",   # neighbor_info
            )

        def on_class_filter(class_value, sel_idx, man_n, knn_n):
            """Handle class filter selection."""
            plot_df = visualizer.get_plot_dataframe(
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=class_value
            )

            if class_value is not None and "class_label" in visualizer.df.columns:
                count = (visualizer.df["class_label"] == class_value).sum()
                status = f"{count} samples"
            else:
                status = ""

            return plot_df, class_value, status

        def on_clear_class(sel_idx, man_n, knn_n):
            """Clear class highlight."""
            plot_df = visualizer.get_plot_dataframe(
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n
            )
            return plot_df, None, None, ""

        def on_model_switch(model_name, _sel_idx, _man_n, _knn_n, _knn_dist, _high_class):
            """Handle model switching (resets all state, inputs unused)."""
            if model_name == visualizer.current_model:
                return (gr.update(),) * 13

            success = visualizer.switch_model(model_name)
            if not success:
                return (gr.update(),) * 13

            plot_df = visualizer.get_plot_dataframe()
            status = f"Showing {len(visualizer.df)} samples ({model_name})"

            return (
                plot_df,                           # umap_plot
                status,                            # status_text
                f"Switched to {model_name}",       # model_status
                None,                              # selected_idx
                [],                                # manual_neighbors
                [],                                # knn_neighbors
                {},                                # knn_distances
                None,                              # highlighted_class
                None,                              # preview_image
                "Click a point to preview",        # preview_details
                None,                              # selected_image
                "Click a point to select",         # selected_details
                visualizer.get_class_options(),    # class_dropdown choices
            )

        def update_neighbor_display(sel_idx, man_n, knn_n, knn_dist):
            """Update neighbor gallery and info with distance info."""
            if sel_idx is None:
                return [], "No neighbors selected"

            man_n = man_n or []
            knn_n = knn_n or []
            knn_dist = knn_dist or {}

            # Combine neighbors: KNN first (sorted by distance), then manual
            all_neighbors = []
            # KNN neighbors sorted by distance
            knn_with_dist = [(idx, knn_dist.get(idx, 999)) for idx in knn_n if idx not in man_n]
            knn_with_dist.sort(key=lambda x: x[1])
            all_neighbors.extend([idx for idx, _ in knn_with_dist])
            # Manual neighbors at end
            all_neighbors.extend(man_n)

            if not all_neighbors:
                return [], "Click points or use Suggest"

            # Build gallery images with distance labels
            images = []
            for idx in all_neighbors[:10]:  # Limit to 10
                if idx < len(visualizer.df):
                    sample = visualizer.df.iloc[idx]
                    img = visualizer.get_image(sample["image_path"])
                    if img is not None:
                        # Build label with class and distance
                        if "class_label" in sample:
                            cls_id = int(sample["class_label"])
                            cls_name = visualizer.get_class_name(cls_id)
                            label = f"{cls_id}: {cls_name}"
                        else:
                            label = f"#{idx}"

                        # Add distance for KNN neighbors
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

        # Wire up events
        # on_load just initializes KNN model, no output needed
        app.load(on_load, outputs=[])

        umap_plot.select(
            on_plot_select,
            inputs=[selected_idx, manual_neighbors, knn_neighbors, highlighted_class],
            outputs=[
                preview_image,
                preview_details,
                selected_image,
                selected_details,
                selected_idx,
                manual_neighbors,
                knn_neighbors,
                umap_plot,
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
            ],
        )

        class_dropdown.change(
            on_class_filter,
            inputs=[class_dropdown, selected_idx, manual_neighbors, knn_neighbors],
            outputs=[umap_plot, highlighted_class, class_status],
        )

        clear_class_btn.click(
            on_clear_class,
            inputs=[selected_idx, manual_neighbors, knn_neighbors],
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
                    preview_image,
                    preview_details,
                    selected_image,
                    selected_details,
                    class_dropdown,
                ],
            )

        # Update neighbor display when neighbors change
        for state in [manual_neighbors, knn_neighbors, knn_distances]:
            state.change(
                update_neighbor_display,
                inputs=[selected_idx, manual_neighbors, knn_neighbors, knn_distances],
                outputs=[neighbor_gallery, neighbor_info],
            )

        # --- Suggest neighbors button ---
        def on_suggest_neighbors(sel_idx, k_val, high_class, man_n):
            """Auto-suggest K nearest neighbors for selected point."""
            if sel_idx is None:
                return gr.update(), [], {}, gr.update(), "Select a point first"

            # Find KNN neighbors
            neighbors = visualizer.find_knn_neighbors(sel_idx, k=int(k_val))
            if not neighbors:
                return gr.update(), [], {}, gr.update(), "No neighbors found"

            # Extract indices and distances
            knn_idx = [idx for idx, _ in neighbors]
            knn_dist = dict(neighbors)

            # Update plot
            plot_df = visualizer.get_plot_dataframe(
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_idx,
                highlighted_class=high_class
            )

            return plot_df, knn_idx, knn_dist, gr.update(), f"Found {len(knn_idx)} neighbors"

        suggest_btn.click(
            on_suggest_neighbors,
            inputs=[selected_idx, knn_k_slider, highlighted_class, manual_neighbors],
            outputs=[umap_plot, knn_neighbors, knn_distances, neighbor_gallery, neighbor_info],
        )

        # --- Clear neighbors button ---
        def on_clear_neighbors(sel_idx, high_class):
            """Clear all neighbors (both manual and KNN)."""
            plot_df = visualizer.get_plot_dataframe(
                selected_idx=sel_idx,
                highlighted_class=high_class
            )
            return plot_df, [], [], {}, [], "No neighbors selected"

        clear_neighbors_btn.click(
            on_clear_neighbors,
            inputs=[selected_idx, highlighted_class],
            outputs=[
                umap_plot, manual_neighbors, knn_neighbors,
                knn_distances, neighbor_gallery, neighbor_info
            ],
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

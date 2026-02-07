"""
GradioVisualizer class for diffviews visualization.

Core visualizer implementing multi-user support, model management,
UMAP plotting, layer caching, and activation extraction.
"""

import json
import os
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import plotly.graph_objects as go

from diffviews.processing.umap import load_dataset_activations
from diffviews.processing.umap_backend import get_knn_class, to_numpy
from diffviews.adapters.registry import get_adapter
from diffviews.core.masking import unflatten_activation

from .models import ModelData
from .gpu_ops import _extract_layer_on_gpu


class GradioVisualizer:
    """Gradio-based visualizer with multi-user support.

    Memory model (single-model-at-a-time):
    - model_configs dict holds paths/config for all discovered models
    - model_data dict holds ONLY the currently loaded model
    - On model switch: unload current → load new (frees ~3GB+ per model)
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

        # R2 cache layer (disabled gracefully if no credentials)
        from diffviews.data.r2_cache import R2LayerCache
        self.r2_cache = R2LayerCache()

        # Shared class labels (typically same across models for ImageNet)
        self.class_labels: Dict[int, str] = {}
        self.load_class_labels()

        # Discover models (populates model_configs)
        self.model_configs: Dict[str, dict] = {}
        self.model_data: Dict[str, ModelData] = {}
        self.discover_models()

        # Store requested initial model for _load_all_models
        self._requested_initial_model = initial_model

        # Load only the initial model (single-model-at-a-time pattern)
        self._load_all_models()

        # Determine default model (for initial UI state)
        if initial_model and initial_model in self.model_configs:
            self.default_model = initial_model
        elif "dmd2" in self.model_configs:
            self.default_model = "dmd2"
        elif self.model_configs:
            self.default_model = list(self.model_configs.keys())[0]
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
        """Load ONLY the initial/default model to minimize memory.

        Single-model-at-a-time pattern: only one model's data + adapter
        in memory. Other models loaded on-demand via _ensure_model_loaded().
        """
        # Determine which model to load initially
        initial = None
        if hasattr(self, '_requested_initial_model') and self._requested_initial_model in self.model_configs:
            initial = self._requested_initial_model
        elif "dmd2" in self.model_configs:
            initial = "dmd2"
        elif self.model_configs:
            initial = next(iter(self.model_configs))

        if initial:
            self._ensure_model_loaded(initial)

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
                    model_data.umap_pca = umap_data.get("pca_reducer")

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

            # Backup default embeddings for restore after layer changes
            model_data.default_df = model_data.df.copy()
            model_data.default_activations = model_data.activations
            model_data.default_umap_reducer = model_data.umap_reducer
            model_data.default_umap_scaler = model_data.umap_scaler
            model_data.default_umap_pca = model_data.umap_pca
            model_data.default_umap_params = dict(model_data.umap_params)
            model_data.default_nn_model = model_data.nn_model

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
        NearestNeighbors = get_knn_class()
        model_data.nn_model = NearestNeighbors(n_neighbors=21, metric="euclidean")
        model_data.nn_model.fit(umap_coords)

    def get_model(self, model_name: str) -> Optional[ModelData]:
        """Get ModelData for a model name, or None if not loaded."""
        return self.model_data.get(model_name)

    def is_valid_model(self, model_name: str) -> bool:
        """Check if a model name is valid (discovered, not necessarily loaded)."""
        return model_name in self.model_configs

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded in memory."""
        return model_name in self.model_data

    def _ensure_model_loaded(self, model_name: str) -> bool:
        """Load a model if not already loaded. Unloads other models first.

        Returns True if model is now loaded, False on error.
        """
        if model_name in self.model_data:
            return True

        if model_name not in self.model_configs:
            print(f"Unknown model: {model_name}")
            return False

        # Unload any currently loaded models first
        for loaded_name in list(self.model_data.keys()):
            self._unload_model(loaded_name)

        # Load the new model
        config = self.model_configs[model_name]
        print(f"Loading model: {model_name}")
        model_data = self._load_model_data(model_name, config)
        if model_data is not None:
            self.model_data[model_name] = model_data
            self.load_adapter(model_name)
            return True
        return False

    def _unload_model(self, model_name: str) -> None:
        """Unload a model from memory, freeing GPU and CPU resources."""
        if model_name not in self.model_data:
            return

        print(f"Unloading model: {model_name}")
        model_data = self.model_data[model_name]

        # Clear adapter from GPU
        if model_data.adapter is not None:
            import torch
            del model_data.adapter
            model_data.adapter = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clear large arrays
        model_data.activations = None
        model_data.default_activations = None
        model_data.df = pd.DataFrame()
        model_data.default_df = None
        model_data.umap_reducer = None
        model_data.umap_scaler = None
        model_data.umap_pca = None
        model_data.nn_model = None
        model_data.default_umap_reducer = None
        model_data.default_umap_scaler = None
        model_data.default_umap_pca = None
        model_data.default_nn_model = None

        # Remove from dict
        del self.model_data[model_name]
        print(f"  Unloaded {model_name}")

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

        # Ensure numpy arrays (cuML returns cupy)
        distances = to_numpy(distances)
        indices = to_numpy(indices)

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

    def get_default_layer_label(self, model_name: str) -> Optional[str]:
        """Get label for pre-computed default embeddings (e.g. 'encoder_bottleneck+midblock')."""
        model_data = self.get_model(model_name)
        if model_data is None:
            return None
        # Use default_umap_params (immutable backup) — umap_params gets
        # overwritten on layer switch and would return the wrong label.
        params = model_data.default_umap_params or model_data.umap_params
        if not params:
            return None
        layers = params.get("layers", [])
        return "+".join(layers) if layers else None

    def get_layer_choices(self, model_name: str) -> List[str]:
        """Get available layer choices: [default_label] + hookable_layers."""
        model_data = self.get_model(model_name)
        if model_data is None:
            return []
        choices = []
        default_label = self.get_default_layer_label(model_name)
        if default_label:
            choices.append(default_label)
        if model_data.adapter is not None:
            for layer in model_data.adapter.hookable_layers:
                if layer not in choices:
                    choices.append(layer)
        return choices

    def _restore_default_embeddings(self, model_name: str):
        """Restore pre-computed default embeddings after a layer change."""
        model_data = self.get_model(model_name)
        if model_data is None or model_data.default_df is None:
            return
        model_data.df = model_data.default_df.copy()
        model_data.activations = model_data.default_activations
        model_data.umap_reducer = model_data.default_umap_reducer
        model_data.umap_scaler = model_data.default_umap_scaler
        model_data.umap_pca = model_data.default_umap_pca
        model_data.umap_params = dict(model_data.default_umap_params) if model_data.default_umap_params else {}
        model_data.nn_model = model_data.default_nn_model
        model_data.current_layer = "default"
        print(f"[{model_name}] Restored default embeddings")

    @staticmethod
    def _clear_layer_data(model_data: "ModelData") -> None:
        """Release heavy layer data from memory before loading a new layer."""
        model_data.activations = None
        model_data.df = pd.DataFrame()
        model_data.umap_reducer = None
        model_data.umap_scaler = None
        model_data.umap_pca = None
        model_data.nn_model = None

    # Max layer cache disk usage per model (bytes). Override via env.
    LAYER_CACHE_MAX_BYTES = int(os.environ.get(
        "DIFFVIEWS_LAYER_CACHE_MAX_MB", "2048"
    )) * 1024 * 1024

    def _evict_layer_cache(self, cache_dir: Path, needed_bytes: int = 300 * 1024 * 1024) -> None:
        """Evict oldest-modified layers until cache_dir is under budget.

        Groups files by layer stem, evicts least-recently-modified first.
        """
        if not cache_dir.exists():
            return

        # Group files by layer name
        layers: dict[str, list[Path]] = {}
        for f in cache_dir.iterdir():
            if f.is_file() and f.suffix in (".csv", ".json", ".npy", ".pkl"):
                layers.setdefault(f.stem, []).append(f)

        if not layers:
            return

        total = sum(f.stat().st_size for files in layers.values() for f in files)
        budget = self.LAYER_CACHE_MAX_BYTES

        if total + needed_bytes <= budget:
            return

        # Sort layers by newest mtime (most recent last = evict from front)
        def layer_mtime(name):
            return max(f.stat().st_mtime for f in layers[name])

        ordered = sorted(layers.keys(), key=layer_mtime)

        for name in ordered:
            if total + needed_bytes <= budget:
                break
            for f in layers[name]:
                sz = f.stat().st_size
                f.unlink()
                total -= sz
            print(f"[cache] Evicted layer {name} from {cache_dir.name}")

    def _load_layer_cache(self, model_name: str, layer_name: str) -> bool:
        """Load cached layer embeddings from disk or R2. Returns True if cache hit."""
        model_data = self.get_model(model_name)
        if model_data is None:
            return False

        cache_dir = model_data.data_dir / "embeddings" / "layer_cache"
        csv_path = cache_dir / f"{layer_name}.csv"
        pkl_path = cache_dir / f"{layer_name}.pkl"
        npy_path = cache_dir / f"{layer_name}.npy"

        # Try R2 download if local cache miss
        if not csv_path.exists() and self.r2_cache.enabled:
            print(f"[{model_name}] Local cache miss, trying R2...")
            self._evict_layer_cache(cache_dir)
            self.r2_cache.download_layer(model_name, layer_name, cache_dir)

        if not csv_path.exists():
            return False

        # Free old layer data before loading new (avoid 2x memory peak)
        self._clear_layer_data(model_data)

        print(f"[{model_name}] Loading cached layer: {layer_name}")
        df = pd.read_csv(csv_path)

        # Load reducer from pkl if available (local-only artifact)
        reducer, scaler, pca_reducer = None, None, None
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                umap_data = pickle.load(f)
            reducer = umap_data["reducer"]
            scaler = umap_data["scaler"]
            pca_reducer = umap_data.get("pca_reducer")

        activations = None
        if npy_path.exists():
            activations = np.load(npy_path, mmap_mode="r")

        # Always refit from cached activations — pkl may be stale or have
        # degenerate .transform() internals (e.g. from n_epochs=0 era).
        # Full fit ensures .transform() works for trajectory projection.
        if activations is not None:
            from diffviews.processing.umap import compute_umap
            print(f"[{model_name}] Fitting UMAP from cached activations...")

            pca_val = os.environ.get("DIFFVIEWS_PCA_COMPONENTS", "50")
            pca_components = None if pca_val.lower() in ("0", "none", "off", "") else int(pca_val)

            embeddings, reducer, scaler, pca_reducer = compute_umap(
                activations, n_neighbors=15, min_dist=0.1,
                normalize=True, pca_components=pca_components,
            )

            # Update cached coordinates to match this fit
            df["umap_x"] = embeddings[:, 0]
            df["umap_y"] = embeddings[:, 1]

            # Save pkl locally for next time
            with open(pkl_path, "wb") as f:
                pickle.dump({"reducer": reducer, "scaler": scaler, "pca_reducer": pca_reducer}, f)
            # Update CSV with new coordinates
            df.to_csv(csv_path, index=False)
            print(f"[{model_name}] Saved refit pkl to {pkl_path}")

        # Load params
        json_path = cache_dir / f"{layer_name}.json"
        umap_params = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                umap_params = json.load(f)

        # Swap into model data
        model_data.df = df
        model_data.activations = activations
        model_data.umap_reducer = reducer
        model_data.umap_scaler = scaler
        model_data.umap_pca = pca_reducer
        model_data.umap_params = umap_params
        self._fit_knn_model(model_data)
        model_data.current_layer = layer_name
        print(f"[{model_name}] Loaded cached layer: {layer_name} ({len(df)} samples)")
        return True

    def extract_layer_activations(
        self, model_name: str, layer_name: str, batch_size: int = 32
    ) -> Optional[np.ndarray]:
        """Extract activations for a single layer across all samples.

        Runs batched forward passes using ActivationExtractor.
        Must be called under _generation_lock (GPU access).

        Returns (N, D) flattened activation matrix, or None on error.
        """
        import torch
        from diffviews.core.extractor import ActivationExtractor

        model_data = self.get_model(model_name)
        if model_data is None or model_data.metadata_df is None:
            return None

        adapter = self.load_adapter(model_name)
        if adapter is None:
            return None

        metadata_df = model_data.metadata_df
        n_samples = len(metadata_df)
        num_classes = adapter.num_classes
        n_batches = (n_samples + batch_size - 1) // batch_size
        print(f"[{model_name}] Extracting {layer_name} for {n_samples} samples ({n_batches} batches)")

        all_activations = []
        extractor = ActivationExtractor(adapter, [layer_name])

        with extractor:
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_meta = metadata_df.iloc[batch_start:batch_end]

                # Load images from PNG paths
                images = []
                sigmas = []
                labels = []
                for _, row in batch_meta.iterrows():
                    img_path = model_data.data_dir / row["image_path"]
                    img = np.array(Image.open(img_path))  # (H, W, 3) uint8
                    images.append(img.transpose(2, 0, 1))  # -> (3, H, W)
                    sigmas.append(row["conditioning_sigma"])
                    labels.append(int(row["class_label"]))

                # Build tensors
                img_tensor = torch.from_numpy(np.stack(images)).float().to(self.device)
                img_tensor = (img_tensor / 127.5) - 1.0
                sigma_tensor = torch.tensor(sigmas, dtype=torch.float32, device=self.device)
                x_noisy = img_tensor * sigma_tensor.view(-1, 1, 1, 1)

                # One-hot class labels
                label_tensor = None
                if num_classes > 0:
                    label_tensor = torch.zeros(len(labels), num_classes, device=self.device)
                    for i, lbl in enumerate(labels):
                        if 0 <= lbl < num_classes:
                            label_tensor[i, lbl] = 1.0

                # Forward pass (hooks capture activations)
                with torch.no_grad():
                    adapter.forward(x_noisy, sigma_tensor, label_tensor)

                # Grab and flatten
                act = extractor.get_activations().get(layer_name)
                if act is not None:
                    if act.dim() > 2:
                        act = act.flatten(1)  # (B, C*H*W)
                    all_activations.append(act.numpy())
                extractor.clear()

                batch_idx = batch_start // batch_size
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx + 1}/{n_batches}")

        if not all_activations:
            return None

        result = np.concatenate(all_activations, axis=0)  # (N, D)
        print(f"[{model_name}] Extracted {layer_name}: shape {result.shape}")
        return result

    def recompute_layer_umap(self, model_name: str, layer_name: str) -> bool:
        """Extract activations for layer, compute UMAP, cache to disk, swap into ModelData.

        Returns True on success, False on error.
        """
        from diffviews.processing.umap import compute_umap, save_embeddings

        model_data = self.get_model(model_name)
        if model_data is None:
            return False

        # Check disk cache first
        if self._load_layer_cache(model_name, layer_name):
            return True

        # Free old layer data before extraction (avoid 2x memory peak)
        self._clear_layer_data(model_data)

        # Extract activations on GPU
        activations = _extract_layer_on_gpu(model_name, layer_name)

        if activations is None:
            print(f"[{model_name}] Failed to extract {layer_name}")
            return False

        # Compute UMAP (CPU, no lock needed)
        # Use PCA pre-reduction if configured (reads DIFFVIEWS_PCA_COMPONENTS env)
        pca_val = os.environ.get("DIFFVIEWS_PCA_COMPONENTS", "50")
        pca_components = None if pca_val.lower() in ("0", "none", "off", "") else int(pca_val)

        print(f"[{model_name}] Computing UMAP for {layer_name} (pca={pca_components})...")
        embeddings, reducer, scaler, pca_reducer = compute_umap(
            activations, n_neighbors=15, min_dist=0.1, normalize=True,
            pca_components=pca_components,
        )

        # Build new df with UMAP coords + original metadata
        new_df = model_data.metadata_df.copy()
        new_df["umap_x"] = embeddings[:, 0]
        new_df["umap_y"] = embeddings[:, 1]

        umap_params = {"layers": [layer_name], "n_neighbors": 15, "min_dist": 0.1}

        # Save to disk cache (evict oldest if over budget)
        cache_dir = model_data.data_dir / "embeddings" / "layer_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._evict_layer_cache(cache_dir)
        csv_path = cache_dir / f"{layer_name}.csv"
        save_embeddings(embeddings, new_df, csv_path, umap_params, reducer, scaler, pca_reducer)
        np.save(cache_dir / f"{layer_name}.npy", activations)
        print(f"[{model_name}] Cached {layer_name} to {cache_dir}")

        # Push to R2 in background (non-blocking)
        self.r2_cache.upload_layer_async(model_name, layer_name, cache_dir)

        # Atomic swap into ModelData
        model_data.df = new_df
        model_data.activations = activations
        model_data.umap_reducer = reducer
        model_data.umap_scaler = scaler
        model_data.umap_pca = pca_reducer
        model_data.umap_params = umap_params
        self._fit_knn_model(model_data)
        model_data.current_layer = layer_name
        return True

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

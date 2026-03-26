# DiffViews Architecture Reference

Comprehensive dependency and API documentation for the DiffViews diffusion model activation visualizer.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File Reference](#file-reference)
   - [Entry Points](#entry-points)
   - [Core Module](#core-module)
   - [Adapters Module](#adapters-module)
   - [Visualization Module](#visualization-module)
   - [Processing Module](#processing-module)
   - [Data Module](#data-module)
   - [Utils Module](#utils-module)
3. [Dependency Graph Summary](#dependency-graph-summary)

---

## Project Overview

DiffViews is an interactive visualization toolkit for exploring diffusion model activations through UMAP embeddings and masked generation. Key features:

- **Model-agnostic adapter interface** for wrapping any diffusion architecture
- **Multi-user Gradio UI** with per-session state
- **HuggingFace Spaces / ZeroGPU deployment** support
- **Single-model-at-a-time memory pattern** (~3GB freed on model switch)
- **Hybrid CPU/GPU mode** via Modal remote workers
- **R2/S3 cloud caching** for layer-specific UMAP embeddings
- **AlignedUMAP 3D visualization** across sigma levels (X,Y = UMAP, Z = log(σ))

---

## File Reference

---

### Entry Points

---

#### `app.py`

**Purpose:** HuggingFace Spaces entry point. Downloads data/checkpoints, initializes visualizer, defines `@spaces.GPU` decorated functions for ZeroGPU.

**Lines:** 433

**Imports from local files:**
- `diffviews.data.r2_cache` → `R2DataStore`
- `diffviews.processing.umap` → `load_dataset_activations`, `compute_umap`, `save_embeddings`
- `diffviews.visualization.app` → `GradioVisualizer`, `create_gradio_app`, `CUSTOM_CSS`, `PLOTLY_HANDLER_JS`
- `diffviews.core.masking` → `ActivationMasker`
- `diffviews.core.generator` → `generate_with_mask_multistep`

**Called by:**
- HuggingFace Spaces runtime (module-level `demo = _setup()`)

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `download_data_r2` | `(output_dir: Path) -> bool` | Download data from Cloudflare R2 | `R2DataStore.download_model_data` | `download_data` |
| `download_data_hf` | `(output_dir: Path) -> None` | Fallback download from HuggingFace Hub | `huggingface_hub.snapshot_download` | `download_data` |
| `download_data` | `(output_dir: Path) -> None` | Download data: R2 first, HF fallback | `download_data_r2`, `download_data_hf` | `ensure_data_ready` |
| `download_checkpoint` | `(output_dir: Path, model: str) -> None` | Download model checkpoint: R2 first, URL fallback | `R2DataStore.download_file`, `urllib.request.urlretrieve` | `ensure_data_ready` |
| `get_pca_components` | `() -> int \| None` | Read PCA pre-reduction setting from env | — | `regenerate_umap` |
| `regenerate_umap` | `(data_dir: Path, model: str) -> bool` | Regenerate UMAP pickle for numba compatibility | `load_dataset_activations`, `compute_umap`, `save_embeddings` | `ensure_data_ready` |
| `check_umap_compatibility` | `(data_dir: Path, model: str) -> bool` | Check if UMAP pickle is compatible with numba | `pickle.load`, `reducer.transform` | `ensure_data_ready` |
| `ensure_data_ready` | `(data_dir: Path, checkpoints: list) -> bool` | Ensure data and checkpoints are downloaded | `download_data`, `download_checkpoint`, `regenerate_umap` | `_setup` |
| `get_device` | `() -> str` | Auto-detect best available device | `torch.cuda.is_available`, `torch.backends.mps.is_available` | `_setup` |
| `generate_on_gpu` | `(model_name, all_neighbors, class_label, n_steps, m_steps, s_max, s_min, guidance, noise_mode, extract_layers, can_project) -> result` | **@spaces.GPU** Run masked generation on GPU | `visualizer.load_adapter`, `visualizer.prepare_activation_dict`, `ActivationMasker`, `generate_with_mask_multistep` | `viz_mod._generate_on_gpu` (injected), Gradio callbacks |
| `extract_layer_on_gpu` | `(model_name, layer_name, batch_size=32) -> np.ndarray` | **@spaces.GPU** Extract layer activations on GPU | `visualizer.extract_layer_activations` | `viz_mod._extract_layer_on_gpu` (injected), Gradio callbacks |
| `_setup` | `() -> gr.Blocks` | Initialize data, visualizer, and Gradio app | `ensure_data_ready`, `GradioVisualizer`, `create_gradio_app` | Module-level execution |

---

### Core Module

---

#### `diffviews/core/extractor.py`

**Purpose:** Activation extraction using PyTorch forward hooks. Captures layer outputs during forward pass.

**Lines:** 197

**Imports from local files:**
- `adapt_diff` → `GeneratorAdapter`

**Called by:**
- `diffviews/core/generator.py`
- `diffviews/visualization/visualizer.py`

---

##### Classes

###### `ActivationExtractor`

**Summary:** Extract activations from a model during forward pass using the adapter hook interface.

| Method | Signature | Summary | Calls | Called By |
|--------|-----------|---------|-------|-----------|
| `__init__` | `(adapter: GeneratorAdapter, layers: List[str] = None)` | Initialize with adapter and target layers | — | `generate_with_mask_multistep`, `GradioVisualizer.extract_layer_activations`, `infer_layer_shape` |
| `_make_hook` | `(name: str) -> Callable` | Create forward hook that stores activations | — | `register_hooks` |
| `register_hooks` | `() -> None` | Register extraction hooks on specified layers | `adapter.register_activation_hooks` | Context manager `__enter__`, explicit calls |
| `remove_hooks` | `() -> None` | Remove all registered hooks | `handle.remove` | Context manager `__exit__`, explicit calls |
| `clear` | `() -> None` | Clear stored activations | — | `generate_with_mask_multistep` |
| `get_activations` | `() -> Dict[str, torch.Tensor]` | Get extracted activations | — | `generate_with_mask_multistep`, `infer_layer_shape` |
| `save` | `(output_path: Path, metadata: Dict = None) -> None` | Save activations to disk as .npz | `np.savez_compressed` | External scripts |

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `flatten_activations` | `(activations: Dict[str, np.ndarray]) -> np.ndarray` | Flatten all layer activations to single vector per sample | `np.concatenate` | `_load_npz_format` in umap.py |
| `load_activations` | `(activation_path: Path) -> Tuple[dict, dict]` | Load activations and metadata from disk | `np.load`, `json.load` | `_load_npz_format` in umap.py |
| `convert_to_fast_format` | `(npz_path: Path, output_path: Path = None, layers: List[str] = None) -> Path` | Convert .npz to fast-loading .npy format | `np.load`, `np.save` | CLI tools |
| `load_fast_activations` | `(npy_path: Path, mmap_mode: str = 'r') -> np.ndarray` | Load pre-concatenated activations with memory mapping | `np.load` | `_load_fast_format` in umap.py |

---

#### `diffviews/core/masking.py`

**Purpose:** Activation masking (replacement) during forward pass. Replaces layer outputs with fixed values.

**Lines:** 183

**Imports from local files:**
- `adapt_diff` → `GeneratorAdapter`

**Called by:**
- `app.py` → `generate_on_gpu`
- `diffviews/visualization/gpu_ops.py`

---

##### Classes

###### `ActivationMasker`

**Summary:** Mask (replace) layer activations with fixed values during forward pass.

| Method | Signature | Summary | Calls | Called By |
|--------|-----------|---------|-------|-----------|
| `__init__` | `(adapter: GeneratorAdapter)` | Initialize with adapter | — | `generate_on_gpu`, `_generate_on_gpu` |
| `set_mask` | `(layer_name: str, activation: torch.Tensor) -> None` | Set fixed activation for a layer | — | `generate_on_gpu`, `_generate_on_gpu` |
| `clear_mask` | `(layer_name: str) -> None` | Remove mask for a layer | — | — |
| `clear_masks` | `() -> None` | Remove all masks | — | — |
| `_make_hook` | `(name: str) -> Callable` | Create forward hook that replaces output | — | `register_hooks` |
| `register_hooks` | `(layers: List[str] = None) -> None` | Register masking hooks | `adapter.register_activation_hooks` | `generate_on_gpu`, `_generate_on_gpu` |
| `remove_hooks` | `() -> None` | Remove all registered hooks | `handle.remove` | `generate_on_gpu`, `_generate_on_gpu` |

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `load_activation_from_npz` | `(npz_path, layer_name: str) -> torch.Tensor` | Load activation from saved NPZ file | `np.load`, `torch.from_numpy` | External usage |
| `compute_mask_dict` | `(activations: np.ndarray, neighbor_indices: List[int], layer_shapes: Dict[str, tuple], layers: List[str] = None) -> Dict[str, np.ndarray]` | Compute mask dict from cached activations (CPU, pure numpy) | `np.mean`, `np.prod` | `_generate_on_gpu` (hybrid mode) |
| `unflatten_activation` | `(flat_activation: torch.Tensor, target_shape: tuple) -> torch.Tensor` | Reshape flattened activation to spatial (1, C, H, W) | `torch.reshape` | `GradioVisualizer.prepare_activation_dict` |

---

#### `diffviews/core/generator.py`

**Purpose:** Image generation with multi-step denoising and optional activation masking.

**Lines:** 373

**Imports from local files:**
- `adapt_diff` → `GeneratorAdapter`
- `diffviews.core.extractor` → `ActivationExtractor`
- `diffviews.core.masking` → `ActivationMasker`

**Called by:**
- `app.py` → `generate_on_gpu`
- `diffviews/visualization/gpu_ops.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `tensor_to_uint8_image` | `(tensor: torch.Tensor) -> torch.Tensor` | Convert tensor [-1,1] to uint8 [0,255] | `torch.clamp`, `torch.permute` | `generate_with_mask`, `generate_with_mask_multistep` |
| `generate_with_mask` | `(adapter, masker, class_label, conditioning_sigma, num_samples, device, seed) -> Tuple` | Generate images with fixed activations (single-step) | `adapter.forward`, `tensor_to_uint8_image` | — |
| `generate_with_mask_multistep` | `(adapter, masker, class_label, num_steps, mask_steps, guidance_scale, num_samples, device, seed, extract_layers, return_trajectory, return_intermediates, return_noised_inputs) -> Tuple` | Multi-step denoising with optional masking | `adapter.get_timesteps`, `adapter.forward_with_cfg`, `adapter.step`, `adapter.decode`, `ActivationExtractor`, `tensor_to_uint8_image` | `generate_on_gpu`, `_generate_on_gpu` |
| `save_generated_sample` | `(image, activations, metadata, output_dir, sample_id) -> Dict` | Save generated image, activations, and metadata | `Image.save`, `np.savez_compressed` | External scripts |
| `infer_layer_shape` | `(adapter: GeneratorAdapter, layer_name: str, device: str = 'cuda') -> Tuple[int, ...]` | Infer activation shape by running dummy forward pass | `adapter.get_layer_shapes`, `ActivationExtractor`, `adapter.forward` | External usage |

---

### Adapters Module (External)

Adapters are provided by the external [`adapt_diff`](https://github.com/mckellcarter/adapt_diff) package.

```python
from adapt_diff import GeneratorAdapter, get_adapter, list_adapters, register_adapter
```

See the adapt_diff repository for adapter documentation and available implementations:
- `dmd2-imagenet-64` - DMD2 ImageNet 64x64 (pixel-space, predicts x0)
- `edm-imagenet-64` - EDM ImageNet 64x64 (pixel-space, predicts x0)
- `mscoco-t2i-128` - MSCOCO Text-to-Image 128x128 (latent-space, predicts ε)
- `abu-custom-sd14` - Custom SD 512x512 (latent-space)

#### Adapter Interface

The unified adapter interface supports both pixel-space (EDM/DMD2) and latent-space (SD-style) models:

```python
class GeneratorAdapter:
    # Model properties
    resolution: int
    num_classes: int
    hookable_layers: List[str]
    prediction_type: str      # "epsilon", "sample", "v_prediction"
    uses_latent: bool         # True for VAE-based models
    in_channels: int          # 3 for pixel, 4 for latent
    conditioning_type: str    # "class", "text", "unconditional"

    # Core forward pass
    def forward(self, x, t, conditioning) -> Tensor
    def forward_with_cfg(self, x, t, cond, uncond, scale) -> Tensor

    # Diffusion schedule and stepping
    def get_timesteps(self, num_steps: int) -> List[Tensor]
    def step(self, x_t, t, pred, t_next=None) -> Tensor
    def get_initial_noise(self, batch_size, device, generator=None) -> Tensor

    # Latent space transforms (identity for pixel-space)
    def encode(self, images: Tensor) -> Tensor
    def decode(self, latent: Tensor) -> Tensor

    # Conditioning
    def prepare_conditioning(self, text=None, class_label=None, ...) -> Any

    # Config and hooks
    def get_default_config(self) -> Dict
    def get_layer_shapes(self) -> Dict[str, Tuple]
```

The generator uses this unified interface:
```python
cond = adapter.prepare_conditioning(text=caption)
timesteps = adapter.get_timesteps(num_steps)
x = adapter.get_initial_noise(batch_size, device)

for i, t in enumerate(timesteps[:-1]):
    pred = adapter.forward_with_cfg(x, t, cond, uncond, guidance_scale)
    x = adapter.step(x, t, pred, t_next=timesteps[i+1])

images = adapter.decode(x)
```

---

### Visualization Module

---

#### `diffviews/visualization/visualizer.py`

**Purpose:** Core GradioVisualizer class implementing multi-user support, model management, UMAP plotting, and activation extraction.

**Lines:** 1228

**Imports from local files:**
- `diffviews.processing.umap` → `load_dataset_activations`
- `diffviews.processing.umap_backend` → `get_knn_class`, `to_numpy`
- `diffviews.processing.aligned_umap` → `compute_aligned_umap`, `load_aligned_embeddings`, `save_aligned_embeddings`, `project_aligned_trajectory_point`
- `adapt_diff` → `get_adapter`
- `diffviews.core.masking` → `unflatten_activation`
- `diffviews.visualization.models` → `ModelData`
- `diffviews.visualization.gpu_ops` → `_extract_layer_on_gpu`

**Called by:**
- `app.py`
- `diffviews/visualization/app.py`

---

##### Classes

###### `GradioVisualizer`

**Summary:** Gradio-based visualizer with multi-user support. Implements single-model-at-a-time memory pattern.

| Method | Signature | Summary | Calls | Called By |
|--------|-----------|---------|-------|-----------|
| `__init__` | `(data_dir, embeddings_path, checkpoint_path, device, num_steps, mask_steps, guidance_scale, sigma_max, sigma_min, label_dropout, adapter_name, max_classes, initial_model)` | Initialize visualizer | `discover_models`, `_load_all_models`, `load_class_labels` | `app.py._setup` |
| `discover_models` | `() -> None` | Discover available models in data directory | — | `__init__` |
| `_load_all_models` | `() -> None` | Load only the initial/default model | `_ensure_model_loaded` | `__init__` |
| `_load_model_data` | `(model_name, config) -> ModelData` | Load all data for a model | `pd.read_csv`, `load_dataset_activations`, `_fit_knn_model` | `_ensure_model_loaded` |
| `_load_activations` | `(data_dir, model_type) -> Tuple` | Load raw activations for generation | `load_dataset_activations` | `_load_model_data` |
| `_fit_knn_model` | `(model_data) -> None` | Fit KNN model on UMAP coordinates | `get_knn_class`, `nn_model.fit` | `_load_model_data`, `_load_layer_cache`, `recompute_layer_umap` |
| `_ensure_umap_loaded` | `(model_data) -> bool` | Lazy load UMAP reducer from pkl | `pickle.load` | `on_generate` callback |
| `get_model` | `(model_name) -> Optional[ModelData]` | Get ModelData for a model | — | Many callbacks |
| `is_valid_model` | `(model_name) -> bool` | Check if model name is valid | — | `on_model_switch` |
| `is_model_loaded` | `(model_name) -> bool` | Check if model is loaded in memory | — | — |
| `_ensure_model_loaded` | `(model_name) -> bool` | Load model if not loaded, unload others | `_unload_model`, `_load_model_data`, `load_adapter` | `_load_all_models`, `on_model_switch` |
| `_unload_model` | `(model_name) -> None` | Unload model, free GPU/CPU resources | `torch.cuda.empty_cache` | `_ensure_model_loaded` |
| `load_class_labels` | `() -> None` | Load ImageNet class labels | `json.load` | `__init__` |
| `get_class_name` | `(class_id) -> str` | Get human-readable class name | — | Many callbacks |
| `find_knn_neighbors` | `(model_name, idx, k, exclude_selected) -> List[Tuple]` | Find k nearest neighbors for a point | `nn_model.kneighbors`, `to_numpy` | `on_suggest_neighbors` |
| `get_image` | `(model_name, image_path) -> Optional[np.ndarray]` | Load image as numpy array | `Image.open` | Callbacks |
| `create_composite_image` | `@staticmethod (main_img, inset_img, inset_ratio, margin, border_width) -> np.ndarray` | Create composite with inset in corner | `Image.paste` | `on_generate` |
| `load_adapter` | `(model_name) -> Optional[Adapter]` | Lazily load generation adapter | `get_adapter`, `AdapterClass.from_checkpoint` | `_ensure_model_loaded`, `generate_on_gpu` |
| `get_default_layer_label` | `(model_name) -> Optional[str]` | Get label for default embeddings | — | Callbacks |
| `get_layer_choices` | `(model_name) -> List[str]` | Get available layer choices | — | Callbacks |
| `_restore_default_embeddings` | `(model_name) -> None` | Restore pre-computed default embeddings | — | `on_layer_change` |
| `_clear_layer_data` | `@staticmethod (model_data) -> None` | Release heavy layer data from memory | — | `_load_layer_cache`, `recompute_layer_umap` |
| `_evict_layer_cache` | `(cache_dir, needed_bytes) -> None` | Evict oldest layers to stay under budget | — | `_load_layer_cache`, `recompute_layer_umap` |
| `_load_layer_cache` | `(model_name, layer_name) -> bool` | Load cached layer embeddings from disk/R2 | `r2_cache.download_layer`, `pd.read_csv`, `compute_umap` | `recompute_layer_umap` |
| `extract_layer_activations` | `(model_name, layer_name, batch_size) -> Optional[np.ndarray]` | Extract activations for layer across all samples | `ActivationExtractor`, `adapter.forward` | `_extract_layer_on_gpu` |
| `recompute_layer_umap` | `(model_name, layer_name) -> bool` | Extract, compute UMAP, cache, swap into ModelData | `_load_layer_cache`, `_extract_layer_on_gpu`, `compute_umap`, `save_embeddings` | `on_layer_change` |
| `prepare_activation_dict` | `(model_name, neighbor_indices) -> Optional[Dict]` | Prepare activation dict for masked generation | `np.mean`, `unflatten_activation` | `generate_on_gpu`, `_generate_on_gpu` |
| `get_plot_dataframe` | `(model_name, selected_idx, manual_neighbors, knn_neighbors, highlighted_class) -> pd.DataFrame` | Get DataFrame for plot with highlight column | — | — |
| `get_class_options` | `(model_name) -> List[Tuple]` | Get class options for dropdown | — | Callbacks |
| `get_color_map` | `(model_name) -> dict` | Generate color map for classes | `matplotlib.pyplot.cm.plasma` | `create_umap_figure` |
| `create_umap_figure` | `(model_name, selected_idx, manual_neighbors, knn_neighbors, highlighted_class, trajectory) -> go.Figure` | Create Plotly UMAP scatter plot (2D or 3D) | `go.Figure`, `go.Scatter`, `go.Scatter3d` | Many callbacks |
| `compute_and_load_aligned_3d` | `(model_name) -> bool` | Compute or load cached AlignedUMAP embeddings | `compute_aligned_umap`, `save_aligned_embeddings`, `_load_aligned_3d` | `on_view_mode_change` |
| `_load_aligned_3d` | `(model_name) -> bool` | Load AlignedUMAP cache from disk | `load_aligned_embeddings` | `compute_and_load_aligned_3d` |
| `_create_3d_figure` | `(model_data, selected_idx, manual_neighbors, knn_neighbors, trajectory) -> go.Figure` | Create Plotly 3D scatter plot with sigma slices | `go.Figure`, `go.Scatter3d` | `create_umap_figure` |
| `project_trajectory_to_3d` | `(model_data, trajectory) -> List[dict]` | Project trajectory points to 3D using k-NN interpolation | `project_aligned_trajectory_point` | `_create_3d_figure` |

---

#### `diffviews/visualization/app.py`

**Purpose:** Gradio app factory. Creates Blocks app with all UI components and event handlers.

**Lines:** 1199

**Imports from local files:**
- `diffviews.utils.device` → `get_device`
- `diffviews.visualization.models` → `ModelData`
- `diffviews.visualization.layout` → `CUSTOM_CSS`, `PLOTLY_HANDLER_JS`
- `diffviews.visualization.gpu_ops` → `_app_visualizer`, `set_visualizer`, `get_visualizer`, `_generate_on_gpu`, `_extract_layer_on_gpu`, etc.
- `diffviews.visualization.visualizer` → `GradioVisualizer`

**Called by:**
- `app.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `create_gradio_app` | `(visualizer: GradioVisualizer) -> gr.Blocks` | Create Gradio Blocks app with all UI components | `set_visualizer`, `gr.Blocks`, event handlers | `app.py._setup` |
| `main` | `() -> None` | CLI entry point for local development | `GradioVisualizer`, `create_gradio_app`, `app.launch` | CLI |

**Key callbacks defined inside `create_gradio_app`:**
- `on_load`, `build_neighbor_gallery`, `on_hover_data`, `on_click_data`
- `on_clear_selection`, `on_class_filter`, `on_clear_class`, `on_model_switch`
- `on_layer_change`, `on_view_mode_change`, `on_suggest_neighbors`, `on_clear_neighbors`
- `on_generate`, `on_clear_generated`, `on_next_frame`, `on_prev_frame`
- `on_gallery_select`, `on_traj_select`

---

#### `diffviews/visualization/models.py`

**Purpose:** ModelData dataclass - per-model data container.

**Lines:** 52

**Imports from local files:** None

**Called by:**
- `diffviews/visualization/visualizer.py`
- `diffviews/visualization/app.py`

---

##### Classes

###### `ModelData`

**Summary:** Per-model data container for thread-safe multi-user access.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Model name |
| `data_dir` | `Path` | Data directory |
| `adapter_name` | `str` | Adapter registry name |
| `checkpoint_path` | `Optional[Path]` | Checkpoint file path |
| `sigma_max` | `float` | Maximum sigma |
| `sigma_min` | `float` | Minimum sigma |
| `default_steps` | `int` | Default denoising steps |
| `df` | `pd.DataFrame` | Embeddings DataFrame |
| `activations` | `Optional[np.ndarray]` | Activation matrix |
| `metadata_df` | `Optional[pd.DataFrame]` | Sample metadata |
| `umap_reducer` | `Any` | Fitted UMAP model |
| `umap_scaler` | `Any` | Fitted StandardScaler |
| `umap_pca` | `Any` | Fitted PCA (if used) |
| `umap_params` | `Dict` | UMAP parameters |
| `umap_pkl_path` | `Optional[Path]` | Path for lazy loading |
| `nn_model` | `Any` | NearestNeighbors model |
| `adapter` | `Any` | Loaded adapter instance |
| `layer_shapes` | `Dict[str, tuple]` | Layer activation shapes |
| `default_*` | Various | Backups for restore after layer change |
| `current_layer` | `str` | Currently active layer |
| `is_3d_mode` | `bool` | Whether in AlignedUMAP 3D mode |
| `sigma_levels` | `List[float]` | Sorted sigma levels (descending) |
| `embeddings_per_sigma` | `Dict[float, np.ndarray]` | Per-sigma UMAP embeddings |
| `nn_models_per_sigma` | `Dict[float, NearestNeighbors]` | Per-sigma KNN models for projection |

---

#### `diffviews/visualization/gpu_ops.py`

**Purpose:** GPU operation wrappers. Supports local and hybrid (Modal remote) modes.

**Lines:** 213

**Imports from local files:**
- `diffviews.core.masking` → `ActivationMasker`, `compute_mask_dict`
- `diffviews.core.generator` → `generate_with_mask_multistep`

**Called by:**
- `diffviews/visualization/app.py`
- `diffviews/visualization/visualizer.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `set_visualizer` | `(visualizer) -> None` | Set global visualizer reference | — | `create_gradio_app` |
| `get_visualizer` | `() -> GradioVisualizer` | Get current visualizer | — | — |
| `set_remote_gpu_worker` | `(worker) -> None` | Set remote GPU worker for hybrid mode | — | Modal setup |
| `is_hybrid_mode` | `() -> bool` | Check if in hybrid CPU/GPU mode | — | — |
| `_generate_on_gpu` | `(model_name, all_neighbors, class_label, n_steps, m_steps, s_max, s_min, guidance, noise_mode, extract_layers, can_project) -> result` | Run masked generation (local or remote) | `compute_mask_dict`, `ActivationMasker`, `generate_with_mask_multistep` | `on_generate` callback |
| `_deserialize_result` | `(result) -> Tuple` | Convert numpy from remote back to torch | `torch.from_numpy` | `_generate_on_gpu` |
| `_deserialize_dict` | `(d) -> Dict` | Convert numpy in dict to torch | `torch.from_numpy` | `_deserialize_result` |
| `_extract_layer_on_gpu` | `(model_name, layer_name, batch_size) -> Optional[np.ndarray]` | Extract layer activations (local only) | `visualizer.extract_layer_activations` | `recompute_layer_umap` |

---

#### `diffviews/visualization/layout.py`

**Purpose:** Layout constants - CSS and JavaScript for Gradio UI.

**Lines:** 468

**Imports from local files:** None

**Called by:**
- `diffviews/visualization/app.py`
- `app.py`

---

##### Constants

| Constant | Description |
|----------|-------------|
| `PLOTLY_HANDLER_JS` | JavaScript for Plotly click/hover event bridge to Gradio textboxes (supports 2D and 3D plots, trajectory hover) |
| `CUSTOM_CSS` | CSS for layout, plot sizing, sidebar width, compact elements |

---

### Processing Module

---

#### `diffviews/processing/umap.py`

**Purpose:** UMAP embedding computation and loading utilities.

**Lines:** 291

**Imports from local files:**
- `diffviews.processing.umap_backend` → `get_umap_class`, `get_backend_name`, `to_numpy`
- `diffviews.core.extractor` → `flatten_activations`, `load_activations`, `load_fast_activations`

**Called by:**
- `app.py`
- `diffviews/visualization/visualizer.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `load_dataset_activations` | `(activation_dir, metadata_path, max_samples, batch_size, low_memory, use_mmap) -> Tuple[np.ndarray, pd.DataFrame]` | Load all activations from dataset (.npy or .npz) | `_load_fast_format`, `_load_npz_format` | `regenerate_umap`, `GradioVisualizer._load_activations` |
| `_load_fast_format` | `(activation_dir, samples, npy_path, use_mmap) -> Tuple` | Load from pre-concatenated .npy file | `load_fast_activations` | `load_dataset_activations` |
| `_load_npz_format` | `(activation_dir, samples, batch_size, low_memory) -> Tuple` | Load from .npz files (slow fallback) | `load_activations`, `flatten_activations` | `load_dataset_activations` |
| `compute_umap` | `(activations, n_neighbors, min_dist, metric, n_components, random_state, normalize, pca_components) -> Tuple[np.ndarray, object, Optional[StandardScaler], Optional[PCA]]` | Compute UMAP with optional PCA pre-reduction | `StandardScaler.fit_transform`, `PCA.fit_transform`, `UMAP.fit_transform` | `regenerate_umap`, `_load_layer_cache`, `recompute_layer_umap` |
| `save_embeddings` | `(embeddings, metadata_df, output_path, umap_params, reducer, scaler, pca_reducer) -> None` | Save UMAP embeddings + metadata to CSV + pkl | `df.to_csv`, `pickle.dump` | `regenerate_umap`, `recompute_layer_umap` |

---

#### `diffviews/processing/umap_backend.py`

**Purpose:** UMAP/KNN backend abstraction - auto-detects cuML GPU vs sklearn CPU.

**Lines:** 68

**Imports from local files:** None

**Called by:**
- `diffviews/processing/umap.py`
- `diffviews/visualization/visualizer.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `_gpu_available` | `() -> bool` | Check if cuML GPU backend is available | `cuml`, `cupy` imports | Module initialization |
| `get_umap_class` | `() -> type` | Return UMAP class (cuML or umap-learn) | — | `compute_umap` |
| `get_aligned_umap_class` | `() -> type` | Return AlignedUMAP class (umap-learn only) | — | `compute_aligned_umap` |
| `get_knn_class` | `() -> type` | Return NearestNeighbors class | — | `_fit_knn_model` |
| `to_gpu_array` | `(data) -> Any` | Convert to cupy array if GPU available | `cp.asarray` | — |
| `to_numpy` | `(data) -> np.ndarray` | Ensure numpy (convert from cupy) | `data.get` | `find_knn_neighbors`, `compute_umap` |
| `get_backend_name` | `() -> str` | Return current backend name for logging | — | `compute_umap` |

---

#### `diffviews/processing/aligned_umap.py`

**Purpose:** AlignedUMAP computation for 3D sigma-varying activation visualization. Creates aligned embeddings across sigma levels where Z = log(sigma).

**Lines:** 284

**Imports from local files:**
- `diffviews.processing.umap_backend` → `get_aligned_umap_class`, `to_numpy`

**Called by:**
- `diffviews/visualization/visualizer.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `compute_aligned_umap` | `(activations, sigma_labels, n_neighbors, min_dist, alignment_regularisation, alignment_window_size, n_components, normalize, pca_components, random_state) -> Tuple[Dict, AlignedUMAP, StandardScaler, PCA, Dict, List, Dict]` | Compute AlignedUMAP across sigma levels with identity relations | `get_aligned_umap_class`, `StandardScaler.fit_transform`, `PCA.fit_transform`, `AlignedUMAP.fit`, `NearestNeighbors.fit` | `GradioVisualizer.compute_and_load_aligned_3d` |
| `project_aligned_trajectory_point` | `(activation, sigma, scaler, pca_reducer, nn_models, embeddings_per_sigma, sigma_levels, k) -> Tuple[float, float]` | Project trajectory point using k-NN interpolation (AlignedUMAP lacks transform) | `nn_model.kneighbors`, `np.average` | `GradioVisualizer.project_trajectory_to_3d` |
| `save_aligned_embeddings` | `(embeddings_per_sigma, metadata_df, output_dir, scaler, pca_reducer, nn_models, sigma_levels, sigma_indices, umap_params) -> None` | Save AlignedUMAP results to disk (CSV, JSON, pkl) | `df.to_csv`, `json.dump`, `pickle.dump` | `compute_and_load_aligned_3d` |
| `load_aligned_embeddings` | `(embeddings_dir) -> Tuple[pd.DataFrame, dict, dict]` | Load AlignedUMAP data from disk | `pd.read_csv`, `json.load`, `pickle.load` | `_load_aligned_3d` |

---

### Data Module

---

#### `diffviews/data/r2_cache.py`

**Purpose:** Cloudflare R2 storage for data hosting and UMAP layer cache.

**Lines:** 289

**Imports from local files:** None

**Called by:**
- `app.py`
- `diffviews/visualization/visualizer.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `_make_r2_client` | `(bucket) -> Tuple[client, bucket, enabled]` | Create boto3 S3 client for R2 | `boto3.client` | `R2DataStore.__init__`, `R2LayerCache.__init__` |

---

##### Classes

###### `R2DataStore`

**Summary:** Bulk data download from R2 for model data hosting.

| Method | Signature | Summary |
|--------|-----------|---------|
| `__init__` | `(bucket: Optional[str])` | Initialize R2 client |
| `enabled` | `@property -> bool` | Whether R2 is available |
| `list_objects` | `(prefix: str) -> list[str]` | List all object keys under prefix |
| `file_exists` | `(key: str) -> bool` | HEAD check on a single key |
| `download_file` | `(key, local_path) -> bool` | Download single file |
| `download_prefix` | `(prefix, local_dir, exclude_dirs, max_workers) -> int` | Download all objects under prefix |
| `download_model_data` | `(model, local_dir) -> bool` | Download all data for a model |

###### `R2LayerCache`

**Summary:** S3-compatible client for layer-specific UMAP cache.

| Method | Signature | Summary |
|--------|-----------|---------|
| `__init__` | `(bucket: Optional[str])` | Initialize R2 client |
| `enabled` | `@property -> bool` | Whether R2 is available |
| `_key` | `(model, layer, ext) -> str` | R2 object key for cache file |
| `layer_exists` | `(model, layer) -> bool` | Check if layer cache exists on R2 |
| `download_layer` | `(model, layer, local_dir) -> bool` | Download layer cache files |
| `upload_layer` | `(model, layer, local_dir) -> bool` | Upload layer cache files |
| `upload_layer_async` | `(model, layer, local_dir) -> None` | Fire-and-forget background upload |

---

#### `diffviews/data/class_labels.py`

**Purpose:** ImageNet class label utilities.

**Lines:** 132

**Imports from local files:** None

**Called by:**
- `diffviews/visualization/visualizer.py` (via direct class label loading)

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `get_data_dir` | `() -> Path` | Get bundled data directory | — | `load_class_labels` |
| `load_class_labels` | `(labels_path) -> Dict[int, str]` | Load ImageNet class labels | `json.load` | `get_class_name` |
| `get_class_name` | `(class_id, labels_path) -> str` | Get class name for ID | `load_class_labels` | External usage |
| `load_class_labels_map` | `(labels_path) -> Dict` | Load full labels map with synsets | `json.load` | — |
| `load_imagenet64_to_standard_mapping` | `(imagenet64_path, standard_path) -> Dict[int, int]` | Load ImageNet64 to standard index mapping | `json.load` | `remap_imagenet64_labels_to_standard` |
| `remap_imagenet64_labels_to_standard` | `(labels) -> np.ndarray` | Remap ImageNet64 labels to standard | `load_imagenet64_to_standard_mapping` | — |

---

### Utils Module

---

#### `diffviews/utils/device.py`

**Purpose:** Device detection and management (CUDA, MPS, CPU).

**Lines:** 83

**Imports from local files:** None

**Called by:**
- `diffviews/visualization/app.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `get_device` | `(prefer_device: str = None) -> str` | Get best available device | `torch.cuda.is_available`, `torch.backends.mps.is_available` | `app.py.main` |
| `get_device_info` | `(device: str) -> dict` | Get information about device | `torch.cuda.get_device_name`, `torch.cuda.memory_allocated` | — |
| `move_to_device` | `(model, device: str) -> model` | Move model to device with MPS fallback | `model.to` | — |

---

### Package Root

---

#### `diffviews/__init__.py`

**Purpose:** Package root. Re-exports core adapter interface.

**Lines:** 19

**Exports:**
- `GeneratorAdapter`
- `get_adapter`
- `list_adapters`
- `register_adapter`

---

## Dependency Graph Summary

```
app.py (entry point)
├── diffviews.data.r2_cache
│   └── R2DataStore, R2LayerCache
├── diffviews.processing.umap
│   ├── load_dataset_activations
│   ├── compute_umap
│   └── save_embeddings
│       └── diffviews.processing.umap_backend
│       └── diffviews.core.extractor
├── diffviews.processing.aligned_umap
│   ├── compute_aligned_umap
│   ├── project_aligned_trajectory_point
│   ├── save_aligned_embeddings
│   └── load_aligned_embeddings
│       └── diffviews.processing.umap_backend
├── diffviews.visualization.app
│   ├── create_gradio_app
│   ├── GradioVisualizer
│   │   ├── diffviews.processing.umap
│   │   ├── diffviews.processing.umap_backend
│   │   ├── diffviews.processing.aligned_umap
│   │   ├── adapt_diff (get_adapter)
│   │   ├── diffviews.core.masking
│   │   └── diffviews.visualization.models
│   │       └── ModelData (includes 3D mode fields)
│   ├── diffviews.visualization.gpu_ops
│   │   ├── diffviews.core.masking
│   │   └── diffviews.core.generator
│   └── diffviews.visualization.layout
│       └── CUSTOM_CSS, PLOTLY_HANDLER_JS (3D support)
├── diffviews.core.masking
│   ├── ActivationMasker
│   └── adapt_diff (GeneratorAdapter)
└── diffviews.core.generator
    ├── generate_with_mask_multistep
    ├── adapt_diff (GeneratorAdapter)
    ├── diffviews.core.extractor
    └── diffviews.core.masking

adapt_diff (external package)
├── GeneratorAdapter (ABC)
├── HookMixin
├── get_adapter()
├── list_adapters()
├── register_adapter()
└── Adapters: dmd2-imagenet-64, edm-imagenet-64, mscoco-t2i-128, abu-custom-sd14
```

---

## Quick Reference for External Use

### Adding a New Model Adapter

See the [adapt_diff repository](https://github.com/mckellcarter/adapt_diff) for creating new adapters.

### Using the Visualizer Programmatically

```python
from adapt_diff import get_adapter

# Get adapter class
AdapterClass = get_adapter('dmd2-imagenet-64')

# Load from checkpoint
adapter = AdapterClass.from_checkpoint('/path/to/checkpoint.pkl', device='cuda')

# Get layer shapes
shapes = adapter.get_layer_shapes()

# Forward pass
output = adapter.forward(x, sigma, class_labels)
```

### Core Generation Pipeline

```python
from diffviews.core.masking import ActivationMasker
from diffviews.core.generator import generate_with_mask_multistep

# Setup masker
masker = ActivationMasker(adapter)
masker.set_mask('encoder_bottleneck', activation_tensor)
masker.register_hooks(['encoder_bottleneck'])

# Generate
images, labels = generate_with_mask_multistep(
    adapter, masker,
    class_label=207,  # golden retriever
    num_steps=5,
    mask_steps=1,
)
```

---

*Generated for diffviews repository architecture documentation.*

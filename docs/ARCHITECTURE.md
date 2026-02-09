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
- `diffviews.adapters.base` → `GeneratorAdapter`

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
- `diffviews.adapters.base` → `GeneratorAdapter`

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
- `diffviews.adapters.base` → `GeneratorAdapter`
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
| `get_denoising_sigmas` | `(num_steps: int, sigma_max: float, sigma_min: float, rho: float = 7.0) -> torch.Tensor` | Generate Karras sigma schedule | `torch.linspace` | `generate_with_mask_multistep` |
| `generate_with_mask` | `(adapter, masker, class_label, conditioning_sigma, num_samples, device, seed) -> Tuple[torch.Tensor, torch.Tensor]` | Generate images with fixed activations (single-step) | `adapter.forward`, `tensor_to_uint8_image` | — |
| `generate_with_mask_multistep` | `(adapter, masker, class_label, num_steps, mask_steps, sigma_max, sigma_min, rho, guidance_scale, stochastic, noise_mode, num_samples, device, seed, extract_layers, return_trajectory, return_intermediates, return_noised_inputs) -> Tuple` | Generate images using multi-step denoising with optional masking | `get_denoising_sigmas`, `ActivationExtractor`, `adapter.forward`, `masker.remove_hooks`, `tensor_to_uint8_image` | `generate_on_gpu`, `_generate_on_gpu` |
| `save_generated_sample` | `(image, activations, metadata, output_dir, sample_id) -> Dict` | Save generated image, activations, and metadata | `Image.save`, `np.savez_compressed` | External scripts |
| `infer_layer_shape` | `(adapter: GeneratorAdapter, layer_name: str, device: str = 'cuda') -> Tuple[int, ...]` | Infer activation shape by running dummy forward pass | `adapter.get_layer_shapes`, `ActivationExtractor`, `adapter.forward` | External usage |

---

### Adapters Module

---

#### `diffviews/adapters/base.py`

**Purpose:** Abstract base class defining the model-agnostic adapter interface.

**Lines:** 159

**Imports from local files:** None

**Called by:**
- All adapter implementations
- `diffviews/core/extractor.py`
- `diffviews/core/masking.py`
- `diffviews/core/generator.py`

---

##### Classes

###### `GeneratorAdapter` (ABC)

**Summary:** Abstract interface for diffusion generator models. Provides model-agnostic access to forward pass, hook registration, and checkpoint loading.

| Property/Method | Signature | Summary |
|-----------------|-----------|---------|
| `model_type` | `@property -> str` | Model identifier string (e.g., 'dmd2-imagenet-64') |
| `resolution` | `@property -> int` | Output image resolution |
| `num_classes` | `@property -> int` | Number of classes (0 for unconditional) |
| `hookable_layers` | `@property -> List[str]` | List of layer names available for hooks |
| `forward` | `(x, sigma, class_labels, **kwargs) -> torch.Tensor` | Forward pass for denoising |
| `register_activation_hooks` | `(layer_names, hook_fn) -> List[RemovableHandle]` | Register forward hooks on layers |
| `get_layer_shapes` | `() -> Dict[str, Tuple[int, ...]]` | Return activation shapes for hookable layers |
| `from_checkpoint` | `@classmethod (checkpoint_path, device, **kwargs) -> GeneratorAdapter` | Load adapter from checkpoint |
| `get_default_config` | `@classmethod () -> Dict[str, Any]` | Return default configuration |
| `to` | `(device: str) -> GeneratorAdapter` | Move model to device |
| `eval` | `() -> GeneratorAdapter` | Set model to eval mode |

---

#### `diffviews/adapters/registry.py`

**Purpose:** Adapter registration, discovery, and lookup by name.

**Lines:** 99

**Imports from local files:**
- `diffviews.adapters.base` → `GeneratorAdapter`

**Called by:**
- Adapter implementations (via `@register_adapter` decorator)
- `diffviews/visualization/visualizer.py`
- `diffviews/__init__.py`

---

##### Functions

| Function | Signature | Summary | Calls | Called By |
|----------|-----------|---------|-------|-----------|
| `register_adapter` | `(name: str) -> Callable` | Decorator to register an adapter class | — | `DMD2ImageNetAdapter`, `EDMImageNetAdapter` |
| `get_adapter` | `(name: str) -> Type[GeneratorAdapter]` | Get adapter class by registered name | `discover_adapters` | `GradioVisualizer.load_adapter` |
| `list_adapters` | `() -> list` | List all registered adapter names | `discover_adapters` | External usage |
| `discover_adapters` | `() -> None` | Auto-discover adapters from entry points | `importlib.metadata.entry_points` | `get_adapter`, `list_adapters` |
| `register_adapter_class` | `(name: str, cls: Type) -> None` | Manually register an adapter class | — | External usage |
| `unregister_adapter` | `(name: str) -> Optional[Type]` | Remove adapter from registry | — | Testing |

---

#### `diffviews/adapters/hooks.py`

**Purpose:** Reusable hook management utilities for adapters.

**Lines:** 111

**Imports from local files:** None

**Called by:**
- `DMD2ImageNetAdapter`
- `EDMImageNetAdapter`

---

##### Classes

###### `HookMixin`

**Summary:** Mixin providing reusable forward hook management for extraction and masking.

| Method | Signature | Summary | Calls | Called By |
|--------|-----------|---------|-------|-----------|
| `__init__` | `() -> None` | Initialize activation/mask/handle storage | — | Adapter `__init__` |
| `make_extraction_hook` | `(layer_name: str) -> Callable` | Create hook that extracts layer output | — | — |
| `make_mask_hook` | `(layer_name: str, mask: torch.Tensor) -> Callable` | Create hook that replaces output with mask | — | — |
| `set_mask` | `(layer_name: str, value: torch.Tensor) -> None` | Store activation mask | — | — |
| `get_mask` | `(layer_name: str) -> Optional[torch.Tensor]` | Get stored mask | — | — |
| `clear_mask` | `(layer_name: str) -> None` | Remove mask for layer | — | — |
| `clear_masks` | `() -> None` | Remove all masks | — | — |
| `get_activations` | `() -> Dict[str, torch.Tensor]` | Return extracted activations | — | — |
| `get_activation` | `(layer_name: str) -> Optional[torch.Tensor]` | Get single activation | — | — |
| `clear_activations` | `() -> None` | Clear all extracted activations | — | — |
| `remove_hooks` | `() -> None` | Remove all registered hooks | — | — |
| `add_handle` | `(handle: RemovableHandle) -> None` | Track hook handle | — | `register_activation_hooks` |
| `num_hooks` | `@property -> int` | Number of active hooks | — | — |

---

#### `diffviews/adapters/dmd2_imagenet.py`

**Purpose:** DMD2 ImageNet 64x64 adapter (single-step distilled model).

**Lines:** 202

**Imports from local files:**
- `diffviews.adapters.base` → `GeneratorAdapter`
- `diffviews.adapters.hooks` → `HookMixin`
- `diffviews.adapters.registry` → `register_adapter`

**Called by:**
- `diffviews/adapters/registry.py` (via decorator registration)

---

##### Classes

###### `DMD2ImageNetAdapter(HookMixin, GeneratorAdapter)`

**Summary:** Adapter for DMD2 ImageNet 64x64 model (EDMPrecond + DhariwalUNet).

Layer naming:
- `encoder_block_N`: N-th encoder block (0-indexed)
- `encoder_bottleneck`: Last encoder block
- `midblock`: First decoder block
- `decoder_block_N`: N-th decoder block

| Method | Signature | Summary |
|--------|-----------|---------|
| `__init__` | `(model, device: str = 'cuda')` | Initialize with loaded model |
| `hookable_layers` | `@property -> List[str]` | Return available layer names |
| `_get_layer_module` | `(layer_name: str) -> nn.Module` | Get PyTorch module for layer name |
| `forward` | `(x, sigma, class_labels, **kwargs) -> torch.Tensor` | Forward pass for denoising |
| `register_activation_hooks` | `(layer_names, hook_fn) -> List[Handle]` | Register hooks on layers |
| `get_layer_shapes` | `() -> Dict[str, Tuple]` | Return activation shapes (runs dummy forward on first call) |
| `from_checkpoint` | `@classmethod (checkpoint_path, device, label_dropout) -> Adapter` | Load from pickle checkpoint |
| `get_default_config` | `@classmethod () -> Dict` | Return default ImageNet 64x64 config |

---

#### `diffviews/adapters/edm_imagenet.py`

**Purpose:** Original EDM ImageNet 64x64 adapter (multi-step iterative sampling).

**Lines:** 316

**Imports from local files:**
- `diffviews.adapters.base` → `GeneratorAdapter`
- `diffviews.adapters.hooks` → `HookMixin`
- `diffviews.adapters.registry` → `register_adapter`

**Called by:**
- `diffviews/adapters/registry.py` (via decorator registration)

---

##### Classes

###### `EDMImageNetAdapter(HookMixin, GeneratorAdapter)`

**Summary:** Adapter for original EDM ImageNet 64x64 model. Unlike DMD2, requires multi-step sampling.

| Method | Signature | Summary |
|--------|-----------|---------|
| `__init__` | `(model, device: str = 'cuda')` | Initialize with loaded model |
| `hookable_layers` | `@property -> List[str]` | Return available layer names |
| `_get_layer_module` | `(layer_name: str) -> nn.Module` | Get PyTorch module for layer name |
| `forward` | `(x, sigma, class_labels, **kwargs) -> torch.Tensor` | Single denoising step |
| `sample` | `(num_samples, class_label, num_steps, sigma_max, sigma_min, rho, S_churn, S_min, S_max, S_noise, device) -> Tuple` | Full EDM sampling with stochastic sampler |
| `register_activation_hooks` | `(layer_names, hook_fn) -> List[Handle]` | Register hooks on layers |
| `get_layer_shapes` | `() -> Dict[str, Tuple]` | Return activation shapes |
| `from_checkpoint` | `@classmethod (checkpoint_path, device, label_dropout) -> Adapter` | Load from pickle checkpoint |

---

### Visualization Module

---

#### `diffviews/visualization/visualizer.py`

**Purpose:** Core GradioVisualizer class implementing multi-user support, model management, UMAP plotting, and activation extraction.

**Lines:** 1228

**Imports from local files:**
- `diffviews.processing.umap` → `load_dataset_activations`
- `diffviews.processing.umap_backend` → `get_knn_class`, `to_numpy`
- `diffviews.adapters.registry` → `get_adapter`
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
| `create_umap_figure` | `(model_name, selected_idx, manual_neighbors, knn_neighbors, highlighted_class, trajectory) -> go.Figure` | Create Plotly UMAP scatter plot | `go.Figure`, `go.Scatter` | Many callbacks |

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
- `on_layer_change`, `on_suggest_neighbors`, `on_clear_neighbors`
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
| `PLOTLY_HANDLER_JS` | JavaScript for Plotly click/hover event bridge to Gradio textboxes |
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
| `get_knn_class` | `() -> type` | Return NearestNeighbors class | — | `_fit_knn_model` |
| `to_gpu_array` | `(data) -> Any` | Convert to cupy array if GPU available | `cp.asarray` | — |
| `to_numpy` | `(data) -> np.ndarray` | Ensure numpy (convert from cupy) | `data.get` | `find_knn_neighbors`, `compute_umap` |
| `get_backend_name` | `() -> str` | Return current backend name for logging | — | `compute_umap` |

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
├── diffviews.visualization.app
│   ├── create_gradio_app
│   ├── GradioVisualizer
│   │   ├── diffviews.processing.umap
│   │   ├── diffviews.processing.umap_backend
│   │   ├── diffviews.adapters.registry
│   │   ├── diffviews.core.masking
│   │   └── diffviews.visualization.models
│   │       └── ModelData
│   ├── diffviews.visualization.gpu_ops
│   │   ├── diffviews.core.masking
│   │   └── diffviews.core.generator
│   └── diffviews.visualization.layout
│       └── CUSTOM_CSS, PLOTLY_HANDLER_JS
├── diffviews.core.masking
│   ├── ActivationMasker
│   └── diffviews.adapters.base
└── diffviews.core.generator
    ├── generate_with_mask_multistep
    ├── diffviews.adapters.base
    ├── diffviews.core.extractor
    └── diffviews.core.masking

diffviews.adapters.registry
├── register_adapter (decorator)
├── get_adapter
├── list_adapters
└── diffviews.adapters.base

diffviews.adapters.dmd2_imagenet
├── @register_adapter('dmd2-imagenet-64')
├── diffviews.adapters.base
├── diffviews.adapters.hooks
└── diffviews.adapters.registry

diffviews.adapters.edm_imagenet
├── @register_adapter('edm-imagenet-64')
├── diffviews.adapters.base
├── diffviews.adapters.hooks
└── diffviews.adapters.registry
```

---

## Quick Reference for External Use

### Adding a New Model Adapter

1. Create `diffviews/adapters/my_model.py`
2. Inherit from `HookMixin` and `GeneratorAdapter`
3. Decorate with `@register_adapter('my-model-name')`
4. Implement required abstract methods

### Using the Visualizer Programmatically

```python
from diffviews import get_adapter

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

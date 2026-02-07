# DiffViews: App Refactoring + cuML Integration Plan

**Branch:** `feature/modal-migrate` (continues M3 work)
**Status:** Planning complete, ready for implementation

---

## Overview

Two integrated goals:
1. **Refactor `app.py`** (~2923 lines) into modules for maintainability and hybrid CPU/GPU architecture
2. **Integrate cuML** for GPU-accelerated UMAP, with pre-seeded layer caches

---

## M4: Cost Optimization (Complete)

Changes already implemented in this branch:

| Change | File | Impact |
|--------|------|--------|
| T4 GPU (was A10G) | `modal_app.py` | ~50% GPU cost reduction |
| `scaledown_window=120s` (was 300s) | `modal_app.py` | Less idle billing |
| Single-model-at-a-time | `diffviews/visualization/app.py` | ~50% memory reduction |

---

## M5: App Refactoring

### Target Structure

```
diffviews/visualization/
├── __init__.py        # Re-exports for backward compatibility
├── models.py          # ModelData dataclass + load/unload functions
├── visualizer.py      # GradioVisualizer class (data access, plotting)
├── gpu_ops.py         # _generate_on_gpu, _extract_layer_on_gpu
├── callbacks.py       # Event handlers from create_gradio_app
├── layout.py          # CUSTOM_CSS, PLOTLY_HANDLER_JS
└── app.py             # create_gradio_app shell, main(), exports
```

### Module Boundaries

#### `models.py` (~150 lines)
```python
@dataclass
class ModelData:
    # Per-model data container (existing)
    ...

def discover_models(root_data_dir: Path) -> Dict[str, dict]:
    """Discover model configs from subdirectories."""

def load_model_data(model_name: str, config: dict, defaults: dict) -> ModelData:
    """Load all data for a model."""

def unload_model(model_data: ModelData) -> None:
    """Release GPU and CPU resources."""

def fit_knn_model(model_data: ModelData) -> None:
    """Fit nearest neighbor model on UMAP coords."""
```

**Dependencies:** numpy, pandas, sklearn, pathlib, json, pickle

#### `visualizer.py` (~800 lines)
```python
class GradioVisualizer:
    """Core visualizer - data access, plotting, layer operations."""

    # Model management
    def __init__(self, data_dir, ...)
    def get_model(self, model_name) -> ModelData
    def is_valid_model(self, model_name) -> bool
    def is_model_loaded(self, model_name) -> bool
    def _ensure_model_loaded(self, model_name) -> bool
    def load_adapter(self, model_name)

    # Data access
    def get_image(self, model_name, image_path)
    def find_knn_neighbors(self, model_name, idx, k)
    def load_class_labels(self)

    # Plotting
    def create_umap_figure(self, model_name, ...)
    def get_plot_dataframe(self, model_name, ...)
    def get_class_options(self, model_name)
    def get_color_map(self, model_name)

    # Layer operations
    def get_layer_choices(self, model_name)
    def _load_layer_cache(self, model_name, layer_name)
    def recompute_layer_umap(self, model_name, layer_name)
    def extract_layer_activations(self, model_name, layer_name)

    # Generation support
    def prepare_activation_dict(self, model_name, ...)
```

**Dependencies:** models.py, gpu_ops.py, plotly, numpy, pandas

#### `gpu_ops.py` (~100 lines)
```python
# Module-level visualizer reference
_app_visualizer = None

def set_visualizer(visualizer):
    """Set global visualizer reference for GPU ops."""
    global _app_visualizer
    _app_visualizer = visualizer

def _generate_on_gpu(model_name, all_neighbors, class_label, ...):
    """GPU generation wrapper. Overridden by HF @spaces.GPU."""
    ...

def _extract_layer_on_gpu(model_name, layer_name, batch_size=32):
    """GPU extraction wrapper. Overridden by HF @spaces.GPU."""
    ...
```

**Dependencies:** diffviews.core.masking, diffviews.core.generator
**Note:** This module is the key for hybrid CPU/GPU split.

#### `callbacks.py` (~400 lines)
```python
from .gpu_ops import _app_visualizer

def get_visualizer():
    """Get current visualizer or raise."""
    if _app_visualizer is None:
        raise RuntimeError("Visualizer not initialized")
    return _app_visualizer

def on_hover_data(hover_json, intermediate_images, model_name):
    visualizer = get_visualizer()
    ...

def on_click_data(click_json, sel_idx, ...):
    ...

def on_model_switch(new_model_name, cur_model, ...):
    ...

def on_layer_change(layer_name, model_name):
    ...

def on_generate(model_name, selected_idx, ...):
    ...

# ... all other event handlers
```

**Dependencies:** gpu_ops.py (for _app_visualizer, _generate_on_gpu), gradio

#### `layout.py` (~500 lines)
```python
PLOTLY_HANDLER_JS = r"""..."""  # Plotly click/hover bridge

CUSTOM_CSS = """..."""  # Styling

def build_ui_components(visualizer) -> dict:
    """Build all Gradio UI components, return as dict."""
    # Optional: extract component building from create_gradio_app
    ...
```

**Dependencies:** gradio

#### `app.py` (~200 lines, refactored)
```python
# Re-exports for backward compatibility
from .models import ModelData
from .visualizer import GradioVisualizer
from .gpu_ops import (
    _app_visualizer,
    _generate_on_gpu,
    _extract_layer_on_gpu,
    set_visualizer,
)
from .layout import CUSTOM_CSS, PLOTLY_HANDLER_JS
from .callbacks import (
    on_hover_data, on_click_data, on_model_switch, ...
)

def create_gradio_app(visualizer: GradioVisualizer) -> gr.Blocks:
    """Create Gradio Blocks app."""
    set_visualizer(visualizer)

    with gr.Blocks(title="DiffViews", css=CUSTOM_CSS, js=PLOTLY_HANDLER_JS) as app:
        # Build UI components
        # Wire event handlers
        ...

    return app

def main():
    """CLI entry point."""
    ...

__all__ = [
    "ModelData", "GradioVisualizer", "create_gradio_app",
    "_app_visualizer", "_generate_on_gpu", "_extract_layer_on_gpu",
    "CUSTOM_CSS", "PLOTLY_HANDLER_JS",
]
```

### Dependency Graph

```
models.py (standalone)
    ↑
visualizer.py ←── gpu_ops.py (uses _app_visualizer global)
    ↑                ↑
callbacks.py ───────┘
    ↑
app.py (wires everything)
    ↑
layout.py (standalone)
```

No circular dependencies.

### Refactoring Order (Tests Pass at Each Step)

1. **Phase 1: Extract models.py**
   - Move `ModelData` dataclass
   - Update imports in app.py
   - Run tests

2. **Phase 2: Extract layout.py**
   - Move `CUSTOM_CSS`, `PLOTLY_HANDLER_JS`
   - Update imports
   - Run tests

3. **Phase 3: Extract gpu_ops.py**
   - Move `_app_visualizer`, `_generate_on_gpu`, `_extract_layer_on_gpu`
   - Add `set_visualizer()` function
   - Update HF `app.py` and `modal_app.py` imports
   - Run tests

4. **Phase 4: Extract visualizer.py**
   - Move `GradioVisualizer` class
   - Import from models.py and gpu_ops.py
   - Run tests

5. **Phase 5: Extract callbacks.py**
   - Move all event handlers from `create_gradio_app`
   - Add `get_visualizer()` helper
   - Run tests

6. **Phase 6: Simplify app.py**
   - Keep only `create_gradio_app` shell and `main()`
   - Add re-exports for backward compatibility
   - Run tests

### Backward Compatibility

All existing imports continue to work:

```python
# These all still work after refactoring:
from diffviews.visualization.app import GradioVisualizer, create_gradio_app
from diffviews.visualization.app import _app_visualizer
from diffviews.visualization.app import CUSTOM_CSS, PLOTLY_HANDLER_JS
from diffviews.visualization.app import ModelData
```

### Test Updates

One test file patch needed:
```python
# Before
patch('diffviews.visualization.app._extract_layer_on_gpu', ...)

# After (or keep both working via re-export)
patch('diffviews.visualization.gpu_ops._extract_layer_on_gpu', ...)
```

---

## M6: cuML GPU Acceleration

### Backend Abstraction

Create `diffviews/processing/umap_backend.py`:

```python
import os
import numpy as np

def _gpu_available() -> bool:
    """Check if cuML GPU backend is available."""
    if os.environ.get("DIFFVIEWS_FORCE_CPU", "").lower() in ("1", "true"):
        return False
    try:
        import cuml
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False

CUML_AVAILABLE = _gpu_available()

def get_umap_class():
    """Return UMAP class (cuML if GPU available, else umap-learn)."""
    if CUML_AVAILABLE:
        from cuml.manifold import UMAP
    else:
        from umap import UMAP
    return UMAP

def get_knn_class():
    """Return KNN class (cuML if GPU available, else sklearn)."""
    if CUML_AVAILABLE:
        from cuml.neighbors import NearestNeighbors
    else:
        from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors

def to_gpu_array(data):
    """Convert to cupy array if GPU available."""
    if CUML_AVAILABLE:
        import cupy as cp
        return cp.asarray(data)
    return np.asarray(data)

def to_numpy(data):
    """Ensure numpy array (convert from cupy if needed)."""
    if hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)
```

### Expected Speedups

| Operation | CPU (umap-learn) | GPU (cuML) | Speedup |
|-----------|------------------|------------|---------|
| UMAP fit (1168 × 50 PCA) | 10-30s | <1s | 30-100x |
| UMAP fit (1168 × 49K raw) | 30s-5min | 1-5s | 10-60x |
| KNN (1168 × 2) | <100ms | <10ms | Already fast |

With cuML, PCA pre-reduction may become unnecessary.

### Modal Image Update

```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        # Existing deps...
        "torch>=2.0.0",
        "numpy>=1.21.0",
        # ...

        # cuML from NVIDIA PyPI
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install("cuml-cu12>=25.02")
    .pip_install("cupy-cuda12x>=12.0")
    .pip_install("diffviews @ git+https://...@main")
)
```

### Pre-Seeding Layer Caches

Since cuML pickles are GPU-specific, pre-seed all common layers to R2:

```bash
# Run on Modal with cuML to generate stable pickles
python scripts/seed_layer_cache.py --execute --layers all

# Uploads to R2:
# data/{model}/layer_cache/{layer}.csv
# data/{model}/layer_cache/{layer}.json
# data/{model}/layer_cache/{layer}.npy
```

With all layers pre-seeded:
- Layer switch = CSV download (no UMAP fit, no GPU needed)
- Trajectory projection = refit from .npy (GPU if cuML, CPU fallback)

### Pickle Portability

cuML pickles are not portable across environments (same issue as umap-learn numba). Current architecture already handles this:

- R2 stores only portable artifacts (.csv, .json, .npy)
- pkl is local-only cache
- Refit from .npy if pkl incompatible

No changes needed — existing `_umap_pkl_ok()` check handles this.

### Environment Fallback Matrix

| Environment | GPU | cuML | Backend |
|-------------|-----|------|---------|
| Modal T4/A10G | Yes | Yes | cuML GPU |
| HF Spaces ZeroGPU | Yes (limited) | No | umap-learn CPU |
| Local dev (CUDA) | Yes | Optional | Auto-detect |
| Local dev (no GPU) | No | No | umap-learn CPU |

---

## M7: Hybrid CPU/GPU Architecture (Future)

After M5 refactoring, the path to hybrid is clear:

### Architecture

```
┌─────────────────────────────────────────┐
│  CPU Container (web)                    │
│  - Gradio UI                            │
│  - UMAP visualization (cached layers)   │
│  - Hover/click handlers                 │
│  - Calls GPU container for heavy ops    │
└──────────────────┬──────────────────────┘
                   │ Modal remote call
                   ▼
┌─────────────────────────────────────────┐
│  GPU Container (worker)                 │
│  - _generate_on_gpu()                   │
│  - _extract_layer_on_gpu()              │
│  - cuML UMAP fit (cache miss)           │
│  - Scales to zero when idle             │
└─────────────────────────────────────────┘
```

### Cost Model

| Scenario | Current (GPU always) | Hybrid |
|----------|---------------------|--------|
| User browsing cached layers | $0.59/hr | $0.02/hr (CPU only) |
| User generates image | $0.59/hr | $0.59/hr + cold start |
| Idle tab open | $0.59/hr | $0.02/hr (CPU only) |

### Implementation (Post-M6)

1. Split `modal_app.py` into `modal_web.py` (CPU) and `modal_gpu.py` (GPU)
2. GPU container exposes `@app.function` for generation/extraction
3. CPU container calls `gpu_function.remote()` for heavy ops
4. Handle cold start UX (loading indicators)

---

## Implementation Checklist

### M4: Cost Optimization ✓
- [x] Switch GPU A10G → T4
- [x] Reduce scaledown_window 300s → 120s
- [x] Single-model-at-a-time loading
- [x] Tests passing (165)

### M5: App Refactoring (Phases 1-4 Complete)
- [x] Phase 1: Extract models.py (ModelData dataclass)
- [x] Phase 2: Extract layout.py (CUSTOM_CSS, PLOTLY_HANDLER_JS)
- [x] Phase 3: Extract gpu_ops.py (_generate_on_gpu, _extract_layer_on_gpu, set_visualizer)
- [x] Phase 4: Extract visualizer.py (GradioVisualizer class)
- [ ] Phase 5: Extract callbacks.py (deferred — tight Gradio coupling)
- [ ] Phase 6: Simplify app.py (deferred)
- [x] Update test patches if needed
- [ ] Verify HF Spaces app.py still works
- [ ] Verify modal_app.py still works

**M5 Results:**
- app.py: 2923 → 1183 lines (60% reduction)
- Tests: 165 passing
- Lint: 8.80/10
- Key architecture (gpu_ops.py) ready for hybrid CPU/GPU split

### M6: cuML Integration
- [x] Create umap_backend.py (auto-detect cuML vs umap-learn/sklearn)
- [x] Update Modal image with cuML deps (cuml-cu12, cupy-cuda12x)
- [x] Update compute_umap() to use backend
- [x] Update _fit_knn_model() to use backend
- [x] Add DIFFVIEWS_FORCE_CPU env var
- [ ] Pre-seed all layer caches to R2
- [ ] Benchmark and document speedups

### M7: Hybrid Architecture
- [x] Split modal_app.py into CPU/GPU (modal_web.py + modal_gpu.py)
- [x] Implement remote GPU function calls (gpu_ops.py set_remote_gpu_worker)
- [x] Add cold start loading UX (Gradio built-in loading)
- [ ] Benchmark cost savings (after deployment)

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `diffviews/processing/umap_backend.py` | 65 | cuML/umap-learn auto-detection |
| `diffviews/visualization/app.py` | 1183 | create_gradio_app + main() + re-exports |
| `diffviews/visualization/visualizer.py` | 1204 | GradioVisualizer class |
| `diffviews/visualization/layout.py` | 467 | CUSTOM_CSS, PLOTLY_HANDLER_JS |
| `diffviews/visualization/gpu_ops.py` | 157 | GPU wrappers + hybrid mode dispatch |
| `diffviews/visualization/models.py` | 51 | ModelData dataclass |
| `modal_gpu.py` | 200 | GPU worker (generation/extraction) |
| `modal_web.py` | 110 | CPU web server (Gradio UI) |
| `modal_app.py` | 318 | Original monolithic Modal entry (with cuML) |
| `diffviews/processing/umap.py` | — | UMAP compute using backend |
| `app.py` (root) | — | HF Spaces entry, injects @spaces.GPU |
| `tests/test_gradio_visualizer.py` | — | 55 tests for visualizer |
| `tests/test_r2_cache.py` | — | 37 tests for R2 operations |

---

## References

- [NVIDIA cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [cuML UMAP Benchmarks](https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/)
- [Modal GPU Pricing](https://modal.com/pricing)
- [Gradio Blocks Documentation](https://www.gradio.app/docs/blocks)

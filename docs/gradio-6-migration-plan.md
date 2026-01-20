# Gradio 6 Migration Plan

## Overview

Migration from Gradio 4.25.0 to Gradio 6.x for HuggingFace Spaces deployment.

**Branch:** `feature/gradio-6-migration`
**Target:** Gradio 6.3.0 (current stable, only maintained version)

## Why Migrate?

Gradio 4 on HF Spaces had multiple issues:
1. `gradio_client` schema generation bug with dict-typed `gr.State`
2. `HfFolder` import error with newer `huggingface_hub`
3. Required fragile monkey-patches
4. Gradio team only maintains version 6.x

## Migration Status

### Completed

- [x] `requirements.txt`: `gradio>=6.0.0`, removed `gradio_client` (bundled)
- [x] `pyproject.toml`: `requires-python>=3.10`, `gradio>=6.0.0`
- [x] `SPACES_README.md`: `sdk_version: 6.3.0`
- [x] Root `app.py`: Removed monkey-patch, added theme/css/head to launch()
- [x] `visualization/app.py`: Moved `CUSTOM_CSS` to module level, relocated Blocks params

### In Progress

- [x] Test locally with Python 3.10+ environment
- [x] Verify click/hover handlers work with Gradio 6
- [x] Verify generation works (vendored torch_utils, see below)
- [x] Regenerate UMAP pickles with current numba
- [ ] Test on HF Spaces

## Breaking Changes Applied

### 1. gr.Blocks Parameters Moved to launch()

**Before (Gradio 4):**
```python
with gr.Blocks(
    title="...",
    theme=gr.themes.Soft(),
    css=custom_css,
    head=f"<script>{JS}</script>",
) as app:
    ...

app.launch(server_name="0.0.0.0")
```

**After (Gradio 6):**
```python
with gr.Blocks(title="...") as app:
    ...

app.launch(
    server_name="0.0.0.0",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
    head=f"<script>{PLOTLY_HANDLER_JS}</script>",
)
```

### 2. CSS Moved to Module Level

CSS is now a module-level constant `CUSTOM_CSS` so it can be accessed from both:
- `diffviews/visualization/app.py:main()`
- Root `app.py` (HF Spaces entry point)

### 3. Python 3.10+ Required

- Gradio 5+ and 6+ require Python >= 3.10
- Updated `pyproject.toml` classifiers to 3.10-3.12
- Updated `tool.black` target-version

### 4. Plotly Handler Persistence

Gradio 6 replaces plot DOM elements on every state update, losing attached event handlers.

**Solution:** MutationObserver + polling to detect element replacement and re-attach handlers.
- Check `isPlotlyReady()` (`.on`, `.data`, `.layout` all exist)
- Track element reference to detect replacement
- Re-attach after MutationObserver detects new nodes

### 5. Hidden Textboxes Must Use CSS

`visible=False` components may not render in DOM in Gradio 6.

**Before:** `gr.Textbox(visible=False)`
**After:** `gr.Textbox(visible=True)` + CSS `#elem-id { display: none; }`

### 6. Gallery Button Parameter

`show_download_button` removed in Gradio 6.

**Before:** `gr.Gallery(show_download_button=False)`
**After:** `gr.Gallery(buttons=[])`

### 7. Scattergl → Scatter (WebGL Context Leak)

Gradio 6 recreates plots on state updates without cleaning up WebGL contexts, causing "Too many active WebGL contexts" warnings.

**Solution:** Use `go.Scatter` (SVG) instead of `go.Scattergl` (WebGL).
- SVG avoids context leak
- Performance equivalent for <10k points
- Consider Scattergl if scaling to 10k+ points (would need manual `Plotly.purge()` cleanup)

### 8. Vendored NVIDIA Modules for Checkpoint Loading

EDM/DMD2 checkpoints are pickles containing class references to `torch_utils` and `dnnlib` from NVIDIA's EDM repo.

**Solution:** Vendor minimal required modules in `diffviews/vendor/`:
```
diffviews/vendor/
├── README.md          # Attribution
├── LICENSE            # CC BY-NC-SA 4.0
├── torch_utils/
│   ├── __init__.py
│   ├── persistence.py # Pickle reconstruction logic
│   └── misc.py        # Utilities (constant, assert_shape, etc.)
└── dnnlib/
    ├── __init__.py
    └── util.py        # EasyDict class
```

Adapters import `ensure_nvidia_modules()` from `diffviews/adapters/nvidia_compat.py` before pickle loading.

## Files Modified

| File | Changes |
|------|---------|
| `requirements.txt` | `gradio>=6.0.0`, removed gradio_client |
| `pyproject.toml` | `requires-python>=3.10`, `gradio>=6.0.0`, classifiers |
| `SPACES_README.md` | `sdk_version: 6.3.0` |
| `app.py` (root) | Removed monkey-patch, import CUSTOM_CSS/PLOTLY_HANDLER_JS, pass to launch() |
| `diffviews/visualization/app.py` | Move CUSTOM_CSS to module level, remove params from Blocks(), Scattergl→Scatter, handler persistence fix, visible textboxes with CSS hiding, Gallery buttons=[] |
| `diffviews/adapters/nvidia_compat.py` | NEW: ensures vendored modules importable |
| `diffviews/adapters/edm_imagenet.py` | Added ensure_nvidia_modules() call |
| `diffviews/adapters/dmd2_imagenet.py` | Added ensure_nvidia_modules() call |
| `diffviews/vendor/` | NEW: vendored torch_utils + dnnlib |
| `data/*/embeddings/*.pkl` | Regenerated with current numba |

## Testing

### Local Testing (requires Python 3.10+)

```bash
# Create environment
conda create -n diffviews310 python=3.10 -y
conda activate diffviews310

# Install dependencies
pip install -r requirements.txt
# or: pip install -e ".[viz]"

# Verify Gradio version
python -c "import gradio; print(f'Gradio {gradio.__version__}')"

# Syntax check
python -m py_compile diffviews/visualization/app.py app.py

# Run app
python -m diffviews.visualization.app --data-dir data
```

### Feature Verification Checklist

- [ ] App loads without errors
- [ ] Plot renders with correct colors
- [ ] Click handler selects points (red ring)
- [ ] Hover handler shows preview
- [ ] KNN suggestion works (lime rings)
- [ ] Manual neighbor toggle works (cyan rings)
- [ ] Gallery displays correctly
- [ ] Class filter highlights points
- [ ] Model switching resets state
- [ ] Generation creates images (if checkpoint available)
- [ ] Trajectory visualization appears
- [ ] Frame navigation works
- [ ] Custom CSS styling preserved

## Remaining Dependency: numba==0.58.1

The `numba` pin is **independent of Gradio** - it's required for UMAP pickle compatibility.

### Why the Pin?

UMAP uses numba for JIT compilation. When you pickle a fitted UMAP reducer, it serializes numba-compiled code that's version-sensitive.

```python
# This pickle contains numba JIT code:
umap_model_path = Path(embeddings_path).with_suffix(".pkl")
with open(umap_model_path, "rb") as f:
    umap_data = pickle.load(f)  # Requires same numba version
```

### Future Options to Remove Pin

1. **Re-generate UMAP pickles** with latest numba
   - Pros: Simple, one-time fix
   - Cons: Must redo when numba updates again

2. **Parametric UMAP** (neural network version)
   - Pros: Saves as PyTorch weights, no numba dependency
   - Cons: Requires TensorFlow or PyTorch backend, different training

3. **Train regression model**
   - Pros: Standard PyTorch, fast inference
   - Cons: Approximate projection, training overhead

4. **Skip trajectory projection** on CPU deployments
   - Pros: No UMAP dependency needed
   - Cons: Loses primary feature

### Recommendation

Keep `numba==0.58.1` for now. Consider parametric UMAP or regression model if:
- Numba updates break pickle compatibility again
- Need cross-platform deployment (UMAP pickles can be OS-specific)

## HuggingFace Spaces Deployment Issues

This section documents issues encountered specifically when deploying to HF Spaces (iframe environment) that didn't appear in local testing.

### Issue 1: Plot Axis Corruption on Click

**When discovered:** First issue on initial HF Spaces test of Gradio 6 port. App loaded successfully, hover worked, first click worked - then subsequent clicks broke the plot.

**Symptom:** First click worked, subsequent clicks caused plot to zoom erratically or "destroy" the view.

**Root Cause:** Plotly's autorange recalculated axis bounds when figure updated with new traces (selection rings, neighbors).

**Attempts:**
1. Added explicit axis ranges with 5% padding in Python:
   ```python
   x_min, x_max = df["umap_x"].min(), df["umap_x"].max()
   x_pad = (x_max - x_min) * 0.05
   fig.update_layout(
       xaxis=dict(range=[x_min - x_pad, x_max + x_pad]),
       yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
   )
   ```
   **Result:** Partially helped but caused other issues.

2. Set `dragmode="pan"` to prevent zoom on drag.
   **Result:** Minor improvement.

3. Added `uirevision=model_name` to preserve zoom/pan state across updates.
   **Result:** Should work but was overridden by explicit ranges.

**Current Solution:** Rely on `uirevision` alone without explicit axis ranges.

### Issue 2: Plot Container Escape

**Symptom:** Plot element moved outside its container, appearing under the right panel columns.

**Root Cause:** Fixed width CSS (800px) conflicted with HF Spaces iframe layout.

**Attempts:**
1. Various `overflow: hidden` and `position: relative` combinations.
   **Result:** Plot became invisible.

2. CSS isolation techniques (`transform: translateZ(0)`, `isolation: isolate`, `contain: layout`).
   **Result:** No effect on container escape.

**Solution:** Use fluid width (100%) with min-height constraints:
```css
#umap-plot {
    min-height: 500px !important;
    height: calc(100vh - 150px) !important;
    flex-grow: 1 !important;
}
```

### Issue 3: Plot Too Small

**Symptom:** Plot occupied only 1/3 of available space.

**Solution:** CSS to expand plot container and ensure Plotly fills it:
```css
#umap-plot > div,
#umap-plot .js-plotly-plot,
#umap-plot .plotly-graph-div {
    height: 100% !important;
    width: 100% !important;
}
```

### Issue 4: Trajectory Not Rendering

**Symptom:** Trajectory generation completed but no trajectory visible on plot. Console showed:
```
[Trajectory] Failed to project step 0: 'Failed in nopython mode pipeline...
```

**Root Cause:** UMAP pickle files contain numba JIT-compiled code that's version-specific. Pre-generated pickles (numba 0.58.1) failed on HF Spaces (different numba version despite pin).

**Why numba pin didn't work:** `requirements.txt` installed diffviews via git first, which pulled its own dependencies before the explicit `numba==0.58.1` line was processed.

**Solution (two-part):**

1. Pin numba in `pyproject.toml` (processed during git install):
   ```toml
   dependencies = [
       ...
       "numba==0.58.1",  # Pin for UMAP pickle compatibility
   ]
   ```

2. Regenerate UMAP on startup if pickle incompatible. Added to `app.py`:
   ```python
   def check_umap_compatibility(data_dir: Path, model: str) -> bool:
       """Check if UMAP pickle is compatible with current numba."""
       # Load pickle, try dummy transform, catch numba errors
       ...

   def regenerate_umap(data_dir: Path, model: str) -> bool:
       """Recompute UMAP from activations, save new pickle."""
       from diffviews.processing.umap import (
           load_dataset_activations,
           compute_umap,
           save_embeddings,
       )
       ...
   ```

   Called during `ensure_data_ready()` after data download.

### Issue 5: Plot View Shift on Interaction (IN PROGRESS)

**Symptom:** Each click on a point or toolbar button causes the plot canvas to shift/pan slightly upward. Accumulates over multiple interactions.

**Root Cause:** Under investigation. Likely related to Gradio 6's DOM replacement behavior in iframe environment.

**Attempts:**

1. **Explicit Python axis ranges** - Set fixed ranges on each figure update.
   **Result:** Didn't prevent shift, may have contributed to it.

2. **CSS isolation** - Various combinations of:
   ```css
   transform: translateZ(0);
   isolation: isolate;
   contain: layout;
   overscroll-behavior: contain;
   ```
   **Result:** No effect.

3. **JavaScript range save/restore** - Save axis ranges before click, restore via `Plotly.relayout()` after DOM mutation:
   ```javascript
   let savedAxisRanges = null;

   function saveAxisRanges() {
       const layout = plotDiv.layout;
       savedAxisRanges = {
           xaxis: [...layout.xaxis.range],
           yaxis: [...layout.yaxis.range]
       };
   }

   function restoreAxisRanges() {
       Plotly.relayout(plotDiv, {
           'xaxis.range': savedAxisRanges.xaxis,
           'yaxis.range': savedAxisRanges.yaxis
       });
   }
   ```
   Called via MutationObserver after Gradio replaces plot DOM.
   **Result:** Didn't prevent shift, possibly fought with uirevision.

4. **Remove explicit ranges, rely on uirevision alone**:
   - Removed Python `xaxis=dict(range=[...])` and `yaxis=dict(range=[...])`
   - Removed JS range save/restore functions
   - Let Plotly's `uirevision=model_name` handle state preservation
   **Result:** Same view shift behavior persists.

**Hypotheses for further investigation:**
- Gradio 6 iframe scrolling behavior
- Plotly modebar interaction triggering container resize
- Hidden element height changes affecting layout
- Need to intercept Gradio's plot update mechanism

### Issue 6: asyncio Cleanup Errors

**Symptom:** Console warnings on shutdown:
```
Exception ignored in: <function BaseEventLoop.__del__...
RuntimeError: Event loop is closed
```

**Root Cause:** Gradio 6 async cleanup in HF Spaces environment.

**Impact:** Cosmetic only, doesn't affect functionality.

**Status:** Not addressed (low priority).

## Phase 5c: JS/CSS Cleanup (IN PROGRESS)

After extensive debugging, accumulated cruft needs cleanup. Fresh analysis identified:

### What to Keep (Matches Clean Pattern)

| Component | Status | Notes |
|-----------|--------|-------|
| JS bridge pattern (hidden textboxes) | ✅ Keep | Core pattern is correct |
| MutationObserver + polling | ✅ Keep | Needed for Gradio DOM replacement |
| Debounced hover with deduplication | ✅ Keep | Prevents flooding |
| `curveNumber === 0` check | ✅ Keep | Ignore overlay traces |
| `visible=True` + CSS hiding | ✅ Keep | Gradio 6 requirement |
| `.change()` not `.input()` | ✅ Keep | Gradio 6 requirement |
| `go.Scatter` not `go.Scattergl` | ✅ Keep | Avoids WebGL leak |
| `uirevision=model_name` | ✅ Keep | View state preservation |

### What to Remove (Accumulated Cruft)

| Component | Lines | Reason |
|-----------|-------|--------|
| `debugDOM()` function | 808-830 | Debug-only, adds noise |
| Extensive retry logging | throughout | Simplify, reduce console spam |
| `attachRetries` + MAX_RETRIES | 942-1006 | Over-engineered, just retry indefinitely |
| `observerSetupRetries` | 1041-1050 | Redundant retry logic |
| Failed CSS isolation attempts | CUSTOM_CSS | `transform: translateZ(0)`, `isolation: isolate`, etc. |

### Cleanup Plan

1. **JS Cleanup** - Remove debug functions, simplify retry logic to infinite retry
2. **CSS Audit** - Remove failed isolation attempts, keep only essential styling
3. **Test on HF** - Verify click/hover still works after cleanup
4. **Investigate Plotly.react()** - Potential fix for view shift (updates in-place vs full replacement)

### Alternative Approach: Plotly.react()

Current: Gradio replaces entire figure on each update.
Alternative: Use `Plotly.react()` to update in-place, may preserve view state better.

```javascript
// Instead of Gradio replacing the plot:
Plotly.react(plotDiv, newData, newLayout, {responsive: true});
```

This would require intercepting Gradio's update mechanism.

## References

- [Gradio 6 Migration Guide](https://www.gradio.app/main/guides/gradio-6-migration-guide)
- [Gradio Changelog](https://www.gradio.app/changelog)
- [UMAP Pickle Issues](https://github.com/lmcinnes/umap/issues/759)
- [Plotly.react() docs](https://plotly.com/javascript/plotlyjs-function-reference/#plotlyreact)

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
- [ ] Verify generation works (requires torch_utils dependency)
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

## Files Modified

| File | Changes |
|------|---------|
| `requirements.txt` | `gradio>=6.0.0`, removed gradio_client |
| `pyproject.toml` | `requires-python>=3.10`, `gradio>=6.0.0`, classifiers |
| `SPACES_README.md` | `sdk_version: 6.3.0` |
| `app.py` (root) | Removed monkey-patch, import CUSTOM_CSS/PLOTLY_HANDLER_JS, pass to launch() |
| `diffviews/visualization/app.py` | Move CUSTOM_CSS to module level, remove params from Blocks(), Scattergl→Scatter, handler persistence fix, visible textboxes with CSS hiding, Gallery buttons=[] |

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

## References

- [Gradio 6 Migration Guide](https://www.gradio.app/main/guides/gradio-6-migration-guide)
- [Gradio Changelog](https://www.gradio.app/changelog)
- [UMAP Pickle Issues](https://github.com/lmcinnes/umap/issues/759)

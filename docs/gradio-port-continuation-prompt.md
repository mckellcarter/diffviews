# Continuation Prompt for DiffViews Visualizer

---

Repo: diffviews
Branch: feature/gradio-6-migration

**Phase 5b: Gradio 6 Migration - Complete (Local Testing)**
**Next: Deploy to HF Spaces**

Start by reading @diffviews/visualization/app.py

---

## What's Working (Gradio 6)

- Plotly scatter plot via `gr.Plot` with `go.Scatter` (SVG, not WebGL)
- Click handling via JS bridge pattern (visible textbox hidden via CSS)
- Hover handling with MutationObserver + polling for handler persistence
- Point selection (red ring highlight)
- Manual neighbor toggling (cyan ring)
- KNN suggestions with distance display (lime ring)
- Neighbor gallery with class labels and distances
- Class filtering, model switching, clear buttons
- **Generation from neighbors** (lazy adapter loading, activation masking) ✅
- **Trajectory visualization** (denoising path on UMAP with sigma labels) ✅
- **Hover preview** (debounced JS hover → preview panel)
- **Intermediate image gallery** (denoising steps with σ labels)
- **Frame navigation** (◀/▶ buttons + gallery click to view intermediate steps)
- **Gallery captions** show full generation info (class ID, class name, step, sigma)
- **Composite images** with noised input inset in upper-left corner
- **Download button** on generated image (Gradio built-in, hidden via `buttons=[]`)
- **CLI command** `diffviews viz` (Gradio-only)
- **CSS styling** compact layouts, vh-based sizing, smooth image scaling
- **Multi-user thread safety** (see Thread Safety section below)

## Hover Preview Implementation

**How it works:**
1. JS `plotly_hover` handler with 150ms debounce writes to hidden `#hover-data-box`
2. `on_hover_data()` loads image and updates preview panel only
3. Click handling separate - updates selection panel, not preview

**Key points:**
- Hover exclusively controls preview panel
- Click controls selection (doesn't touch preview)
- Model switch clears preview (data is new)
- Only main data trace (curve 0) triggers hover

## Intermediate Gallery Implementation

**How it works:**
1. `on_generate()` calls with `return_intermediates=True`
2. Generator returns `(images, labels, trajectory, intermediates)`
3. Each step image displayed in gallery with σ label

**UI:**
- Gallery below generated image: 5 columns, 80px height
- Labels show σ value at each step
- Cleared by "Clear" button along with trajectory

## Trajectory Implementation

**How it works:**
1. `on_generate()` calls `generate_with_mask_multistep()` with `return_trajectory=True` and `extract_layers`
2. Each step's activations are projected through `umap_reducer.transform()`
3. `create_umap_figure()` renders trajectory as:
   - Lime green dashed line connecting points
   - Green gradient markers (light→medium green)
   - Star marker for start, diamond for end

**State management:**
- `trajectory_coords = gr.State(value=[])` stores list of trajectories `[[(x,y,σ),...], ...]`
- Multiple trajectories accumulate with each generation
- Trajectory preserved through: selection changes, neighbor toggles, class filter
- Trajectory cleared only by: Clear Generated button, model switch

## Frame Navigation Implementation

**How it works:**
1. ◀/▶ buttons step through intermediate images
2. Clicking gallery thumbnail shows that step in main generated_image panel
3. `generation_info` state stores class/step metadata for captions
4. Gallery caption shows: `{class_id}: {class_name} | Step {i}/{n} | σ={sigma}`

**State:**
- `intermediate_images = gr.State(value=[])` - list of `(img, sigma)` tuples
- `animation_frame = gr.State(value=-1)` - current frame (-1 = final)
- `generation_info = gr.State(value=None)` - `{class_id, class_name, n_traj, n_steps}`

## Thread Safety (Multi-User Support)

**Architecture:**
- `ModelData` dataclass holds all per-model data (df, activations, nn_model, umap_reducer, adapter)
- All models preloaded at init into `model_data` dict - **read-only after init**
- No mutable `current_model` on visualizer - model selection is per-session via `gr.State`
- All methods take `model_name` as first parameter
- All event handlers receive `current_model` from gr.State

**Verified behaviors:**
- Sessions in different browser tabs maintain independent state
- Model switch in one tab doesn't affect another tab
- Selection, neighbors, trajectories isolated per-session
- Generation uses correct model per-session

**Key classes/methods:**
- `ModelData` (dataclass, lines 29-56)
- `_load_all_models()` - preloads all models at init
- `get_model(model_name)` - returns ModelData or None
- `is_valid_model(model_name)` - validation helper
- `current_model = gr.State(value=visualizer.default_model)` in app

## Remaining Items

### Phase 5b: Gradio 6 Migration (COMPLETE)

**Why Gradio 6:**
- Gradio 4 had multiple compatibility issues on HF Spaces (schema bugs, HfFolder import)
- Required monkey-patches that were fragile
- Gradio 6 is the only maintained version

**Completed:**
- [x] Update `requirements.txt`: `gradio>=6.0.0`
- [x] Update `pyproject.toml`: `requires-python>=3.10`, `gradio>=6.0.0`
- [x] Update `SPACES_README.md`: `sdk_version: 6.3.0`
- [x] Remove monkey-patch from root `app.py`
- [x] Migrate `visualization/app.py`: move `theme/css/js` from `gr.Blocks()` to `launch()`
- [x] Move `CUSTOM_CSS` to module level constant
- [x] Fix hidden textboxes (`visible=True` + CSS hiding)
- [x] Fix Gallery buttons (`buttons=[]` replaces `show_download_button`)
- [x] Fix Plotly handler persistence (MutationObserver + polling)
- [x] Fix WebGL context leak (`go.Scatter` instead of `go.Scattergl`)
- [x] Test locally with Python 3.10 - click/hover working

**Completed (continued):**
- [x] Verify generation works (vendored torch_utils/dnnlib for checkpoint loading)
- [x] Regenerate UMAP pickles with current numba (fixes trajectory projection)

**In Progress:**
- [ ] Test on HF Spaces free tier (CPU)
- [ ] Test on HF Spaces paid tier (GPU)

**Vendored NVIDIA Modules:**
EDM/DMD2 checkpoints are pickles containing class references to `torch_utils` and `dnnlib`.
Now vendored in `diffviews/vendor/` under CC BY-NC-SA 4.0:
- `diffviews/vendor/torch_utils/` - persistence.py, misc.py
- `diffviews/vendor/dnnlib/` - util.py (EasyDict)
- `diffviews/adapters/nvidia_compat.py` - ensure_nvidia_modules() helper

**Gradio 6 Breaking Changes Applied:**

| Change | Before (Gradio 4) | After (Gradio 6) |
|--------|-------------------|------------------|
| Blocks params | `gr.Blocks(theme=, css=, head=)` | `gr.Blocks(title=)` then `launch(theme=, css=, js=)` |
| CSS location | Inline in function | Module-level `CUSTOM_CSS` constant |
| Hidden components | `visible=False` | `visible=True` + CSS `display:none` |
| Gallery download | `show_download_button=False` | `buttons=[]` |
| Textbox events | `.input()` | `.change()` |
| Plot updates | Handlers persist | Need MutationObserver + polling |
| Python version | >=3.8 | >=3.10 |

**Current Requirements:**
```
gradio>=6.0.0
numba==0.58.1  # UMAP pickle compatibility
huggingface_hub>=0.19.0
python_version: "3.10"  # in Space README
```

**UMAP Pickle Issue (RESOLVED):**
UMAP pickles were regenerated with current numba version. Trajectory projection now works.
The `numba==0.58.1` pin may no longer be strictly required but kept for stability.
If issues recur, options include:
1. Re-generate UMAP pickles with current numba ✅ (done)
2. Use parametric UMAP (saves as PyTorch weights)
3. Train regression model to approximate projection

See `docs/gradio-6-migration-plan.md` for full migration details.

### Phase 6: Auth & Persistence (DEFERRED)
See `docs/auth-plan.md` for comprehensive plan covering:
- Basic auth CLI flags, HF OAuth, FastAPI OAuth
- SQLite persistence layer for generations/workspaces
- Usage logging and analytics
- Security considerations and deployment-specific options

---

## JS Bridge Pattern (for reference)

**Key learnings (Gradio 6):**
1. Inject JS via `js=` param in `launch()` (not `head=`)
2. Use `.change()` event on textbox (not `.input()`)
3. Textboxes must be `visible=True` with CSS hiding (visible=False not in DOM)
4. Use `go.Scatter` (SVG) to avoid WebGL context leak
5. Need MutationObserver + polling to re-attach handlers after plot updates

**Pattern (Gradio 6):**
```python
# Module-level constants:
PLOTLY_HANDLER_JS = """..."""
CUSTOM_CSS = """#click-data-box, #hover-data-box { display: none; }..."""

# In gr.Blocks (minimal params):
with gr.Blocks(title="...") as app:
    # visible=True but hidden via CSS - Gradio 6 doesn't render visible=False
    click_data_box = gr.Textbox(value="", elem_id="click-data-box", visible=True)
    # ... rest of UI

# In launch() (theme/css/js):
app.launch(
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
    js=PLOTLY_HANDLER_JS,  # Note: js= not head=
)

# JS attaches plotly_click handler with MutationObserver for persistence:
# (in PLOTLY_HANDLER_JS)
plotDiv.on('plotly_click', function(data) {
    clickBox.value = JSON.stringify({pointIndex: point.customdata, ...});
    clickBox.dispatchEvent(new Event('change', { bubbles: true }));
});
// MutationObserver re-attaches when Gradio replaces plot element

# Python handler wired to .change():
click_data_box.change(handler, inputs=[click_data_box, ...], outputs=[...])
```

---

## Current State

- `app.py` (~2200 lines) with Plotly + generation + hover + intermediates + thread safety
- 41 tests passing (visualizer + generator)
- Phase 5b complete (local testing), ready for HF Spaces deployment

**Key files:**
- `diffviews/visualization/app.py` - main Gradio app
- `diffviews/scripts/cli.py` - CLI with `viz` command
- `diffviews/core/generator.py` - generation functions
- `diffviews/adapters/nvidia_compat.py` - vendored module loader
- `diffviews/vendor/` - vendored torch_utils + dnnlib
- `tests/test_gradio_visualizer.py` - visualizer tests
- `tests/test_generator.py` - generator tests

```bash
# Run tests
python -m pytest tests/test_gradio_visualizer.py tests/test_generator.py -v

# Run via CLI (recommended)
diffviews viz --data-dir data

# Run directly
python -m diffviews.visualization.app --data-dir data
```

---

## Generation Architecture

**Activation flow (critical - order matters!):**
1. Activations stored as `(N, D)` flattened array where D = sum of layer sizes
2. Layers concatenated in **sorted** order during UMAP training
3. `prepare_activation_dict()` splits back by layer using same sorted order
4. Each layer activation reshaped to `(1, C, H, W)` for masking

**Key methods (all take model_name as first param):**
- `load_adapter(model_name)`: Lazy loads checkpoint, caches layer shapes in model_data
- `prepare_activation_dict(model_name, neighbor_indices)`: Averages neighbors, splits by layer
- `on_generate(..., model_name)`: Combines neighbors, creates masker, generates image, extracts trajectory
- `create_umap_figure(model_name, ..., trajectory=)`: Renders denoising path on plot
- `find_knn_neighbors(model_name, idx, k)`: KNN search using model's nn_model
- `get_image(model_name, path)`: Load image from model's data_dir

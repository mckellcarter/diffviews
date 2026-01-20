# Continuation Prompt for DiffViews Visualizer

---

Repo: diffviews
Branch: feature/phase5-hf-spaces-deployment

**Phase 5: HuggingFace Spaces Deployment**

Start by reading @diffviews/visualization/app.py

---

## What's Working

- Plotly scatter plot via `gr.Plot` with `go.Scattergl`
- Click handling via JS bridge pattern (hidden textbox)
- Point selection (red ring highlight)
- Manual neighbor toggling (cyan ring)
- KNN suggestions with distance display (lime ring)
- Neighbor gallery with class labels and distances
- Class filtering, model switching, clear buttons
- **Generation from neighbors** (lazy adapter loading, activation masking)
- **Trajectory visualization** (denoising path on UMAP with sigma labels)
- **Hover preview** (debounced JS hover → preview panel)
- **Intermediate image gallery** (denoising steps with σ labels)
- **Frame navigation** (◀/▶ buttons + gallery click to view intermediate steps)
- **Gallery captions** show full generation info (class ID, class name, step, sigma)
- **Composite images** with noised input inset in upper-left corner
- **Download button** on generated image (Gradio built-in, hidden on intermediate gallery)
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

### Phase 5: HF Spaces Deployment (IN PROGRESS)

**Completed:**
- [x] Create HF Spaces `app.py` entry point in repo root
- [x] Add `requirements.txt` for Spaces compatibility
- [x] Create Space on HuggingFace Hub (mckell/diffviews)
- [x] Configure data/checkpoint auto-download on startup
- [x] Add data existence checks (config + embeddings + images)

**In Progress:**
- [ ] Fix gradio_client schema generation bug
- [ ] Test on HF Spaces free tier (CPU)
- [ ] Test on HF Spaces paid tier (GPU)
- [ ] Add deployment documentation

**Key Issues & Solutions:**

1. **Python 3.13 pickle/numba incompatibility**
   - Error: `TypeError: code() argument 13 must be str, not int`
   - Solution: Pin `python_version: "3.10"` in Space README YAML header

2. **numba version mismatch with UMAP pickle**
   - Error: `Dispatcher._rebuild() got unexpected keyword argument 'impl_kind'`
   - Solution: Pin `numba==0.58.1` in requirements.txt

3. **Gradio 6 HfFolder import error**
   - Error: `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`
   - Solution: Pin `huggingface_hub>=0.19.0,<0.28.0`

4. **gradio_client additionalProperties bug**
   - Error: `TypeError: argument of type 'bool' is not iterable`
   - Cause: `knn_distances = gr.State(value={})` creates dict schema with `additionalProperties: true`
   - Solution: Monkey-patch `get_type()` in app.py to return "Any" for boolean schemas

**Requirements Pins (critical):**
```
numba==0.58.1
gradio==4.25.0
gradio_client==0.15.0
huggingface_hub>=0.19.0,<0.28.0
python_version: "3.10"  # in Space README
```

**Files for Manual Upload to Space:**
- `app.py` - Entry point with auto-download + monkey-patch
- `requirements.txt` - Pinned dependencies
- `README.md` (from SPACES_README.md) - Space metadata

### Phase 6: Auth & Persistence (DEFERRED)
See `docs/auth-plan.md` for comprehensive plan covering:
- Basic auth CLI flags, HF OAuth, FastAPI OAuth
- SQLite persistence layer for generations/workspaces
- Usage logging and analytics
- Security considerations and deployment-specific options

---

## JS Bridge Pattern (for reference)

**Key learnings:**
1. Inject JS via `head=` param in `gr.Blocks()` (not `js=` on events, not `gr.HTML`)
2. Use `.input()` event on textbox (not `.change()`)
3. Include textbox in inputs list explicitly - Gradio doesn't auto-pass
4. Use `go.Scattergl` with `.tolist()` for Gradio compatibility
5. Benign "too many arguments" warning can be ignored

**Pattern:**
```python
# In gr.Blocks:
head=f"<script>{CLICK_HANDLER_JS}</script>"

# Hidden textbox:
click_data_box = gr.Textbox(value="", elem_id="click-data-box", visible=False)

# JS attaches plotly_click handler, writes JSON to textbox:
CLICK_HANDLER_JS = """
plotDiv.on('plotly_click', function(data) {
    clickBox.value = JSON.stringify({pointIndex: point.customdata, ...});
    clickBox.dispatchEvent(new Event('input', { bubbles: true }));
});
"""

# Python handler wired to .input():
click_data_box.input(handler, inputs=[click_data_box, ...], outputs=[...])
```

---

## Current State

- `app.py` (~2200 lines) with Plotly + generation + hover + intermediates + thread safety
- 33 Gradio tests + 24 generator tests (57 total)
- Phase 4 complete, ready for deployment

**Key files:**
- `diffviews/visualization/app.py` - main Gradio app
- `diffviews/scripts/cli.py` - CLI with `viz` command
- `diffviews/core/generator.py` - generation functions
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

# Continuation Prompt for Gradio Port

---

I'm continuing a Gradio port of a Dash visualization app.

Repo: diffviews
Branch: feature/gradio-port-phase4-polish

**Phase 4 IN PROGRESS: Polish Features**

Start by reading @diffviews/visualization/gradio_app.py

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

## Remaining Phase 4 Items

Potential improvements:
1. Loading indicators during generation
2. Export generated images

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

- `gradio_app.py` (~1500 lines) with Plotly + generation + hover + intermediates
- 30 Gradio tests + 24 generator tests (54 total)
- Phase 4 in progress

**Key files:**
- `diffviews/visualization/gradio_app.py` - main app
- `diffviews/visualization/app.py` - Dash reference
- `diffviews/core/generator.py` - generation functions
- `tests/test_gradio_visualizer.py` - gradio tests
- `tests/test_generator.py` - generator tests (includes gradio generation tests)

```bash
python -m pytest tests/test_gradio_visualizer.py tests/test_generator.py -v
python -m diffviews.visualization.gradio_app --data-dir data
```

---

## Generation Architecture

**Activation flow (critical - order matters!):**
1. Activations stored as `(N, D)` flattened array where D = sum of layer sizes
2. Layers concatenated in **sorted** order during UMAP training
3. `prepare_activation_dict()` splits back by layer using same sorted order
4. Each layer activation reshaped to `(1, C, H, W)` for masking

**Key methods:**
- `load_adapter()`: Lazy loads checkpoint, caches layer shapes
- `prepare_activation_dict(neighbor_indices)`: Averages neighbors, splits by layer
- `on_generate()`: Combines neighbors, creates masker, generates image, extracts trajectory
- `create_umap_figure(..., trajectory=)`: Renders denoising path on plot

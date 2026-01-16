# Continuation Prompt for Gradio Port

---

I'm continuing a Gradio port of a Dash visualization app.

Repo: diffviews
Branch: feature/gradio-port-phase3-generation

**Phase 3b COMPLETE: Trajectory Visualization Working**

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

## Trajectory Implementation

**How it works:**
1. `on_generate()` calls `generate_with_mask_multistep()` with `return_trajectory=True` and `extract_layers`
2. Each step's activations are projected through `umap_reducer.transform()`
3. `create_umap_figure()` renders trajectory as:
   - White dotted line connecting points
   - Viridis-colored markers with sigma labels
   - Star marker for start (high noise), diamond for end (low noise)

**State management:**
- `trajectory_coords = gr.State(value=[])` stores `[(x, y, sigma), ...]`
- Trajectory preserved when: toggling neighbors, changing class filter, suggesting KNN
- Trajectory cleared when: selecting new point, clearing selection, switching model

## Next: Phase 4 (Polish)

Potential improvements:
1. Hover preview on UMAP points
2. Intermediate image gallery (denoising steps)
3. Loading indicators during generation
4. Export generated images
5. Trajectory animation option

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

- `gradio_app.py` (~1350 lines) with Plotly + generation + trajectory working
- 30 Gradio tests + 24 generator tests (54 total)
- Next: Phase 4 (polish)

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

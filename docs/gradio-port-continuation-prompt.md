# Continuation Prompt for Gradio Port

---

I'm continuing a Gradio port of a Dash visualization app.

Repo: diffviews
Branch: feature/gradio-port-phase2-selection

**Phase 3 COMPLETE: Generation Working**

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

## Next: Trajectory Visualization (Phase 3b)

Need to show denoising trajectory on UMAP plot:

1. Call `generate_with_mask_multistep()` with `return_trajectory=True, extract_layers=[...]`
2. Project each step's activations through `umap_reducer.transform()`
3. Add trajectory trace to Plotly figure (line + markers with sigma labels)

**Key code locations:**
- `on_generate()` handler: ~line 1161 in gradio_app.py
- `generate_with_mask_multistep()`: diffviews/core/generator.py
- UMAP reducer loaded from `.pkl` file (stored in `visualizer.umap_reducer`)

**Implementation sketch:**
```python
# Modify on_generate() to extract trajectory:
images, labels, trajectory_acts, intermediates = generate_with_mask_multistep(
    ...,
    extract_layers=sorted(visualizer.umap_params.get("layers", [])),
    return_trajectory=True,
    return_intermediates=True,
)

# Project through UMAP (if reducer available):
if visualizer.umap_reducer and trajectory_acts:
    trajectory_coords = []
    for act in trajectory_acts:
        if visualizer.umap_scaler:
            act = visualizer.umap_scaler.transform(act)
        coords = visualizer.umap_reducer.transform(act)
        trajectory_coords.append((float(coords[0, 0]), float(coords[0, 1])))

    # Store in state for plot update
    # Add go.Scatter trace with mode="lines+markers+text"
```

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

- `gradio_app.py` (~1230 lines) with Plotly + generation working
- 28 Gradio tests + 24 generator tests (52 total)
- Next: Phase 3b (trajectory) or Phase 4 (polish)

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

**Key methods added:**
- `load_adapter()`: Lazy loads checkpoint, caches layer shapes
- `prepare_activation_dict(neighbor_indices)`: Averages neighbors, splits by layer
- `on_generate()`: Combines neighbors, creates masker, generates image

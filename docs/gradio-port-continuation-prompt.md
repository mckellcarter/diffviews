# Continuation Prompt for Gradio Port

Copy this prompt to continue work on the Gradio port in a new session:

---

I'm continuing a Gradio port of a Dash visualization app.

Repo: diffviews
Branch: feature/gradio-port-phase2-selection (in progress)
Next: feature/gradio-port-phase3-generation

Phase 1 done: Basic layout, ScatterPlot, model switching, class filtering.
Phase 2 done: KNN suggest button, neighbor gallery with distances, 23 tests pass.

Phase 3 TODO: Enable generation from averaged neighbor activations.

See docs/gradio-port-plan.md for full context.
Start by reading @diffviews/visualization/gradio_app.py

---

## Detailed Context (if needed)

**What exists:**
- `diffviews/visualization/gradio_app.py` (~890 lines) - Gradio app with:
  - `GradioVisualizer` class: data loading, model discovery, KNN fitting
  - `find_knn_neighbors(idx, k)` returns (idx, distance) tuples
  - ScatterPlot with click selection, class filtering
  - Suggest button + K slider for auto-suggesting neighbors
  - Gallery shows distances (d=X.XX) for KNN neighbors
  - Generation controls disabled (placeholders)

- `tests/test_gradio_visualizer.py` - 23 tests
- 87 total tests pass

**Key technical notes:**
- Using `gr.ScatterPlot` (Altair) not Plotly - has working `.select()` events
- Per-session state: `selected_idx`, `manual_neighbors`, `knn_neighbors`, `knn_distances`
- Thread lock scaffolded for generation: `_generation_lock`

**Phase 3 needs:**
- Enable generate button when checkpoint available
- Port `generate_from_neighbors` logic from Dash app
- Progress indicator during generation
- Save generated images to session temp dir

```bash
# Run tests
python -m pytest tests/test_gradio_visualizer.py -v

# Run app
python -m diffviews.visualization.gradio_app --data-dir data
```

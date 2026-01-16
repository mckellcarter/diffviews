# Gradio Port Implementation Plan

## Overview

Port the Dash visualization app (`diffviews/visualization/app.py`) to Gradio for:
- Better multi-user support (built-in session state, queue)
- Easier deployment (HuggingFace Spaces, Modal)
- Future TransformerLens integration (Gradio has strong ML/interp community adoption)

## Branch Structure

```
main (production-ready Dash app)
 └── feature/gradio-port (integration branch)
      ├── feature/gradio-port-phase1-core    ✅ COMPLETE
      ├── feature/gradio-port-phase2-selection (next)
      ├── feature/gradio-port-phase3-generation
      └── feature/gradio-port-phase4-polish
```

## Phase Status

### Phase 1: Core Layout ✅ COMPLETE
**Branch:** `feature/gradio-port-phase1-core`
**PR:** → `feature/gradio-port`

Completed:
- [x] Add gradio dependency to pyproject.toml
- [x] Create `gradio_app.py` with `GradioVisualizer` class
- [x] Port data loading and model discovery from Dash app
- [x] Implement basic Blocks layout (3-column: sidebar, plot, controls)
- [x] Add ScatterPlot with native click selection
- [x] Add model switching (dropdown + state reset)
- [x] Add class filtering (dropdown + highlight)
- [x] Generation controls as placeholders (disabled)
- [x] Fix checkpoint path in test_dmd2_adapter.py
- [x] Add comprehensive tests (19 tests in test_gradio_visualizer.py)

Files created/modified:
- `diffviews/visualization/gradio_app.py` (new, ~750 lines)
- `tests/test_gradio_visualizer.py` (new, 19 tests)
- `pyproject.toml` (added gradio optional dependency)
- `tests/test_dmd2_adapter.py` (fixed checkpoint path)

### Phase 2: Selection & Neighbors (IN PROGRESS)
**Branch:** `feature/gradio-port-phase2-selection`

Completed:
- [x] Add KNN-based automatic neighbor suggestions (`find_knn_neighbors` + Suggest button)
- [x] Enhance neighbor gallery display with distance info (shows d=X.XX for KNN, "manual" for clicked)
- [x] Add Clear Neighbors button
- [x] Add K slider for adjusting neighbor count
- [x] Store neighbor distances in session state (`knn_distances`)
- [x] Add 4 new tests for `find_knn_neighbors` (23 total tests now)
- [x] Add plasma colormap for class colors

**BLOCKER: ScatterPlot limitations require Plotly switch**

ScatterPlot (Altair) issues discovered:
- Cannot control plot size (internal Vega-Lite constraints)
- `.select()` returns coordinates `[x, y]`, not row indices
- Zoom triggers broken select events

**TODO: Switch to Plotly with JS bridge**
1. Add `create_umap_figure()` method using Plotly
2. Use hidden Textbox + JavaScript `plotly_click` handler for click events
3. Replace `gr.ScatterPlot` with `gr.Plot`
4. Update all callbacks to return Plotly figures

Reference: Dash app `app.py` lines 802-816 (Plotly figure), 962-1005 (clickData handling)

### Phase 3: Generation
**Branch:** `feature/gradio-port-phase3-generation`

TODO:
- [ ] Enable generation button when checkpoint available
- [ ] Port `generate_from_neighbors` logic with thread-safe locking
- [ ] Add progress indicator during generation
- [ ] Save generated images to session-specific temp directory
- [ ] Update plot with generated sample trajectory
- [ ] Add "Clear Generated" functionality

Key considerations:
- Thread safety: Use `threading.Lock` for shared adapter (already scaffolded)
- Per-session state: Generated samples in `gr.State`, not shared DataFrame
- Trajectory visualization: May need Plotly for line traces

### Phase 4: Polish & Production
**Branch:** `feature/gradio-port-phase4-polish`

TODO:
- [ ] Add hover preview (may need custom JS or click-to-preview)
- [ ] Improve CSS styling (match Dash Bootstrap look)
- [ ] Add authentication option for deployment
- [ ] Configure queue settings for concurrent users
- [ ] Test multi-user scenarios
- [ ] Add deployment documentation (HuggingFace Spaces, Modal)
- [ ] Performance optimization for large datasets

## Architecture Decisions

### Why ScatterPlot over Plot?
- `gr.ScatterPlot` has native `.select()` event for click handling
- `gr.Plot` (Plotly) doesn't expose click events in Gradio 4.x
- Trade-off: Less visual customization, but working interactions

### State Management
- Per-session: `gr.State` for selected_idx, neighbors, highlighted_class
- Shared: visualizer.df, visualizer.activations (read-only after init)
- Thread-safe: `_generation_lock` for adapter access during generation

### File Structure
```
diffviews/visualization/
    app.py          # Original Dash app (keep for now)
    gradio_app.py   # New Gradio app
```

## Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Run Gradio-specific tests:
```bash
python -m pytest tests/test_gradio_visualizer.py -v
```

## Local Development

```bash
# Install with gradio support
pip install -e ".[gradio]"

# Run Gradio app
python -m diffviews.visualization.gradio_app --data-dir data --port 7860

# Run with share link (for testing)
python -m diffviews.visualization.gradio_app --data-dir data --share
```

## Deployment Targets

1. **HuggingFace Spaces** (recommended for demos)
   - Free GPU tier available
   - Easy sharing with ML community

2. **Modal** (recommended for production)
   - Serverless GPU, scales to zero
   - Good for TransformerLens integration (large models)

3. **Self-hosted** (Gunicorn + nginx)
   - Full control
   - Requires GPU server management

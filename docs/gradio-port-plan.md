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

### Phase 2: Selection & Neighbors ✅ COMPLETE
**Branch:** `feature/gradio-port-phase2-selection`

Completed:
- [x] Add KNN-based automatic neighbor suggestions (`find_knn_neighbors` + Suggest button)
- [x] Enhance neighbor gallery display with distance info (shows d=X.XX for KNN, "manual" for clicked)
- [x] Add Clear Neighbors button
- [x] Add K slider for adjusting neighbor count
- [x] Store neighbor distances in session state (`knn_distances`)
- [x] Add 4 new tests for `find_knn_neighbors` (23 total tests now)
- [x] Add plasma colormap for class colors
- [x] **Switch from ScatterPlot to Plotly with JS bridge**

**Plotly Migration (completed):**
- Replaced `gr.ScatterPlot` with `gr.Plot` + `go.Scattergl`
- Added `create_umap_figure()` method for Plotly figure generation
- Implemented JS bridge pattern for click handling (hidden textbox + plotly_click)
- Updated all callbacks to return Plotly figures

**JS Bridge Key Learnings:**
1. Inject JS via `head=` param in `gr.Blocks()` (not `js=` on events)
2. Use `.input()` event on textbox (not `.change()`)
3. Include textbox in inputs list explicitly
4. Use `go.Scattergl` with `.tolist()` for Gradio compatibility
5. Benign "too many arguments" warning can be ignored

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
- [ ] Add hover preview using JS bridge pattern (see below)
- [ ] Improve CSS styling (match Dash Bootstrap look)
- [ ] Add authentication option for deployment
- [ ] Configure queue settings for concurrent users
- [ ] Test multi-user scenarios
- [ ] Add deployment documentation (HuggingFace Spaces, Modal)
- [ ] Performance optimization for large datasets

## Architecture Decisions

### Why Plotly with JS Bridge (updated)
Initially used `gr.ScatterPlot` but hit blockers:
- ScatterPlot `.select()` returns coordinates, not row indices
- Cannot control plot size
- Zoom triggers broken events

**Solution:** `gr.Plot` with Plotly + custom JS bridge
- JS `plotly_click` handler writes click data to hidden textbox
- Textbox `.input()` event triggers Python callback
- Full control over plot appearance and interactions
- `customdata` field carries row indices for click handling

### Hover Preview Implementation (Phase 4)
Use same JS bridge pattern with debounce:
```javascript
let hoverTimeout;
plotDiv.on('plotly_hover', function(data) {
    clearTimeout(hoverTimeout);
    hoverTimeout = setTimeout(() => {
        hoverBox.value = JSON.stringify({pointIndex: point.customdata});
        hoverBox.dispatchEvent(new Event('input', { bubbles: true }));
    }, 150);  // debounce prevents flooding backend
});
plotDiv.on('plotly_unhover', () => clearTimeout(hoverTimeout));
```
- Add hidden `hover_data_box` textbox
- Python handler updates `preview_image` and `preview_details` only
- Keep click handler separate for selection

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

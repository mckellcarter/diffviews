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
      ├── feature/gradio-port-phase1-core        ✅ COMPLETE
      ├── feature/gradio-port-phase2-selection   ✅ COMPLETE
      ├── feature/gradio-port-phase3-generation  ✅ COMPLETE (merged)
      └── feature/gradio-port-phase4-polish      (next)
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
- `diffviews/visualization/gradio_app.py` (new, ~1400 lines)
- `tests/test_gradio_visualizer.py` (new, 30 tests)
- `tests/test_generator.py` (24 tests including gradio generation)
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

### Phase 3: Generation ✅ COMPLETE
**Branch:** `feature/gradio-port-phase2-selection` (combined with Phase 2)

Completed:
- [x] Enable generation button when checkpoint available
- [x] Port `generate_from_neighbors` logic with thread-safe locking
- [x] Add `load_adapter()` method for lazy adapter loading
- [x] Add `prepare_activation_dict()` to split activations by layer (sorted order critical!)
- [x] Wire up generation settings (steps, mask_steps, guidance, sigma_max/min)
- [x] Display generated image in UI
- [x] Add 5 new tests for generation methods in `test_generator.py`

**Phase 3b - Trajectory ✅ COMPLETE:**
- [x] Add trajectory visualization on UMAP plot
- [x] Add "Clear Generated" button
- [x] Multiple trajectories accumulate until cleared
- [x] Trajectory persists through selection/neighbor changes
- [ ] Show intermediate images during denoising (deferred to Phase 4)

**Trajectory Implementation (completed):**
- Lime green dashed line connecting points
- Green gradient markers (light→medium) with sigma in hover
- Star marker for start, diamond for end
- Clicks on trajectory points ignored (only curve 0 handled)
- State: `trajectory_coords = [[(x,y,σ),...], ...]` (list of trajectories)

Key considerations:
- Thread safety: Use `threading.Lock` for shared adapter ✅ implemented
- Per-session state: Generated samples in `gr.State`, not shared DataFrame
- UMAP reducer must be loaded (from .pkl file) for trajectory projection

### Phase 4: Polish & Production ✅ COMPLETE
**Branch:** `feature/gradio-port-phase4-polish`

Completed:
- [x] Add hover preview using JS bridge pattern (debounced plotly_hover → preview panel)
- [x] Show intermediate images during denoising (gallery with σ labels)
- [x] Frame navigation (◀/▶ buttons + gallery click to view steps)
- [x] Gallery captions show full generation info (class, step, sigma)
- [x] Trajectory hover shows intermediate images in preview panel
- [x] Composite images with noised input inset in upper-left
- [x] Download button on generated image (Gradio built-in)
- [x] Hide download button on intermediate gallery (prevents caption overlap)
- [x] Add `diffviews viz-gradio` CLI command (alongside existing `viz` for Dash)
- [x] CSS styling (compact layouts, vh-based sizing, smooth image scaling)
- [x] **Multi-user thread safety refactoring** (see below)
- [x] Configure queue settings (`max_size=20`)
- [x] Test multi-user scenarios (verified session isolation)

### Phase 5: Public Demo (CURRENT)

Focus: Minimal functional public deployment without auth

TODO:
- [ ] Add deployment documentation (HuggingFace Spaces)
- [ ] Create `app.py` for HF Spaces (standard entry point)
- [ ] Add `requirements.txt` for Spaces compatibility
- [ ] Test on HF Spaces free tier
- [ ] Performance optimization for large datasets (if needed)

### Phase 6: Auth & Persistence (DEFERRED)

**Plan:** See `docs/auth-plan.md` for comprehensive auth + persistence design

Deferred items (implement when needed for private/enterprise deployments):
- [ ] Basic auth CLI flags (`--auth`, `--auth-file`)
- [ ] HuggingFace OAuth integration
- [ ] User persistence layer (SQLite + file storage)
- [ ] Generation history, saved workspaces
- [ ] Usage logging/analytics

**Thread Safety Refactoring (completed):**
- Added `ModelData` dataclass for per-model data isolation
- All models preloaded at init into `model_data` dict (read-only after init)
- Replaced mutable `current_model` with `gr.State` for per-session model selection
- Updated all methods to accept `model_name` parameter
- Updated all event handlers to pass `current_model` state
- 33 tests including `TestMultiUserIsolation` class

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

### State Management (Thread-Safe Multi-User)
- **Per-session** via `gr.State`:
  - `current_model` - which model this session is using
  - `selected_idx`, `manual_neighbors`, `knn_neighbors`, `knn_distances`
  - `highlighted_class`, `trajectory_coords`
  - `intermediate_images`, `animation_frame`, `generation_info`
- **Shared read-only** (populated at init, never modified):
  - `visualizer.model_data[model_name]` - ModelData instances with df, activations, nn_model, umap_reducer
  - `visualizer.class_labels` - shared label mapping
- **Thread-safe mutation**:
  - `_generation_lock` protects lazy adapter loading
  - Adapter stored per-model in `model_data[name].adapter`

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

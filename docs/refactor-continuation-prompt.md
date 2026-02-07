# Continuation Prompt: App Refactoring + cuML Integration

Paste this after clearing context to resume work.

---

## Prompt

I'm refactoring the diffviews visualizer for better modularity and cuML GPU acceleration. The full plan is in `docs/refactor-and-cuml-plan.md` — read it first.

### What's Done

**M3 (Modal compute):** Complete, merged to main (PR #76). Modal serverless GPU deployment working.

**M4 (Cost optimization):** Complete on `feature/modal-migrate` branch:
- T4 GPU (was A10G) — ~50% cost reduction
- `scaledown_window=120s` (was 300s)
- Single-model-at-a-time loading in GradioVisualizer
- `_ensure_model_loaded()`, `_unload_model()` methods added
- Tests: 165 passing, lint 8.84/10

### What's Next

**M5 (App refactoring):** Break `diffviews/visualization/app.py` (2923 lines) into modules:

```
diffviews/visualization/
├── models.py      # ModelData dataclass + load/unload
├── visualizer.py  # GradioVisualizer class
├── gpu_ops.py     # _generate_on_gpu, _extract_layer_on_gpu
├── callbacks.py   # Event handlers
├── layout.py      # CUSTOM_CSS, PLOTLY_HANDLER_JS
└── app.py         # create_gradio_app shell + re-exports
```

**Refactoring order (tests pass at each step):**
1. Extract `models.py` (ModelData dataclass)
2. Extract `layout.py` (CSS/JS constants)
3. Extract `gpu_ops.py` (GPU wrappers — key for hybrid)
4. Extract `visualizer.py` (GradioVisualizer class)
5. Extract `callbacks.py` (event handlers)
6. Simplify `app.py` (shell + re-exports)

**M6 (cuML integration):** After refactoring:
- Create `diffviews/processing/umap_backend.py` (auto-detect cuML vs umap-learn)
- Update Modal image with `cuml-cu12`, `cupy-cuda12x`
- Pre-seed all layer caches to R2 with cuML-generated pickles
- Add `DIFFVIEWS_FORCE_CPU` env var for fallback

### Key Architecture Notes

- `gpu_ops.py` is the critical module for hybrid CPU/GPU architecture
- `_app_visualizer` global set by `set_visualizer()` in `create_gradio_app`
- Callbacks access visualizer via `get_visualizer()` helper
- All existing imports preserved via re-exports in `app.py`
- cuML pickles are GPU-specific; R2 stores only portable artifacts (.csv, .json, .npy)

### Current Status

**Branch:** `feature/modal-migrate`
**Tests:** 165 passing
**Changed files (M4):**
- `modal_app.py` — T4 GPU, scaledown_window
- `diffviews/visualization/app.py` — single-model loading
- `tests/test_gradio_visualizer.py` — updated tests

### Commands

```bash
# Run tests
python -m pytest tests/ -v

# Lint
pylint diffviews/visualization/app.py modal_app.py --disable=C0114,C0115,C0116,R0913,R0914,R0915,R0912,R0902,W0612,W0611,W0718,W1514,E0401 --max-line-length=120

# Local Modal test
modal serve modal_app.py

# Deploy
modal deploy modal_app.py
```

Please read the plan file and continue from where we left off.

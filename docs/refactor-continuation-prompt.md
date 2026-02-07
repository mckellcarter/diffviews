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

**M5 (App refactoring):** Phases 1-4 complete on `feature/modal-migrate` branch:

```
diffviews/visualization/
├── models.py      # ModelData dataclass (51 lines)
├── layout.py      # CUSTOM_CSS, PLOTLY_HANDLER_JS (467 lines)
├── gpu_ops.py     # _generate_on_gpu, _extract_layer_on_gpu, set_visualizer (76 lines)
├── visualizer.py  # GradioVisualizer class (1204 lines)
├── callbacks.py   # NOT extracted (tight Gradio coupling)
└── app.py         # create_gradio_app + main() + re-exports (1183 lines)
```

**M5 Results:**
- app.py: 2923 → 1183 lines (60% reduction)
- Tests: 165 passing
- Lint: 8.80/10
- Key architecture (`gpu_ops.py`) ready for hybrid CPU/GPU split

### What's Next

**M6 (cuML integration):**
- Create `diffviews/processing/umap_backend.py` (auto-detect cuML vs umap-learn)
- Update Modal image with `cuml-cu12`, `cupy-cuda12x`
- Pre-seed all layer caches to R2 with cuML-generated pickles
- Add `DIFFVIEWS_FORCE_CPU` env var for fallback

**M7 (Hybrid CPU/GPU):** After M6:
- Split `modal_app.py` into CPU web + GPU worker
- CPU handles UI, cached layer viz; GPU only for generation/extraction
- Significant cost savings (GPU scales to zero when idle)

### Key Architecture Notes

- `gpu_ops.py` is the critical module for hybrid CPU/GPU architecture
- `_app_visualizer` global set by `set_visualizer()` in `create_gradio_app`
- Callbacks access visualizer via `get_visualizer()` helper
- All existing imports preserved via re-exports in `app.py`
- cuML pickles are GPU-specific; R2 stores only portable artifacts (.csv, .json, .npy)

### Current Status

**Branch:** `feature/modal-migrate`
**Tests:** 165 passing
**Lint:** 8.80/10

**New files (M5):**
- `diffviews/visualization/models.py` — ModelData dataclass
- `diffviews/visualization/layout.py` — CSS/JS constants
- `diffviews/visualization/gpu_ops.py` — GPU wrappers
- `diffviews/visualization/visualizer.py` — GradioVisualizer class

**Modified files:**
- `diffviews/visualization/app.py` — now just create_gradio_app + main + re-exports
- `tests/test_gradio_visualizer.py` — updated patch paths for gpu_ops

### Commands

```bash
# Run tests
python -m pytest tests/ -v

# Lint all visualization modules
pylint diffviews/visualization/*.py --disable=C0114,C0115,C0116,R0913,R0914,R0915,R0912,R0902,W0612,W0611,W0718,W1514,E0401 --max-line-length=120

# Local Modal test
modal serve modal_app.py

# Deploy
modal deploy modal_app.py
```

Please read the plan file and continue with M6 (cuML integration).

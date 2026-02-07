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
├── gpu_ops.py     # GPU wrappers + hybrid dispatch (157 lines)
├── visualizer.py  # GradioVisualizer class (1204 lines)
├── callbacks.py   # NOT extracted (tight Gradio coupling)
└── app.py         # create_gradio_app + main() + re-exports (1183 lines)
```

**M6 (cuML integration):** Complete on `feature/modal-migrate` branch:
- `diffviews/processing/umap_backend.py` — auto-detect cuML vs umap-learn/sklearn
- `umap.py` uses `get_umap_class()` + `to_numpy()` for portable output
- `visualizer.py` uses `get_knn_class()` for KNN
- `modal_app.py` has cuML deps (`cuml-cu12>=25.02`, `cupy-cuda12x>=12.0`)
- `DIFFVIEWS_FORCE_CPU=1` env var for fallback

**M7 (Hybrid CPU/GPU):** Complete on `feature/modal-migrate` branch:
- `modal_gpu.py` — GPU worker with `GPUWorker` class (generate, extract_layer, compute_umap)
- `modal_web.py` — CPU web server, calls GPU worker via `modal.Cls.lookup()`
- `gpu_ops.py` — `set_remote_gpu_worker()` for hybrid dispatch

### What's Next

**Before merging to main:**
- Revert `@feature/modal-migrate` → `@main` in modal_app.py, modal_gpu.py, modal_web.py

**Remaining tasks:**
- Pre-seed all layer caches to R2 with cuML-generated embeddings
- Benchmark cost savings (hybrid vs monolithic)
- Deploy and test hybrid architecture

### Key Architecture Notes

- `gpu_ops.py` supports both local and remote GPU execution
- `set_remote_gpu_worker(worker)` enables hybrid mode
- `is_hybrid_mode()` checks if remote worker is configured
- cuML auto-detected at import; `DIFFVIEWS_FORCE_CPU=1` forces CPU backend
- R2 stores portable artifacts (.csv, .json, .npy); pkl is local-only

### Current Status

**Branch:** `feature/modal-migrate`
**Tests:** 165 passing
**Lint:** 8.31/10

**New files (M6/M7):**
- `diffviews/processing/umap_backend.py` — cuML/sklearn auto-detection
- `modal_gpu.py` — GPU worker for hybrid architecture
- `modal_web.py` — CPU web server for hybrid architecture

**Modified files:**
- `diffviews/processing/umap.py` — uses backend for UMAP
- `diffviews/visualization/gpu_ops.py` — hybrid dispatch support
- `diffviews/visualization/models.py` — generic KNN type
- `diffviews/visualization/visualizer.py` — uses backend for KNN
- `modal_app.py` — cuML deps added

### Commands

```bash
# Run tests
python -m pytest tests/ -v

# Lint
pylint diffviews/visualization/*.py diffviews/processing/umap_backend.py --disable=C0114,C0115,C0116,R0913,R0914,R0915,R0912,R0902,W0612,W0611,W0718,W1514,E0401,C0415 --max-line-length=120

# Deploy monolithic (simpler, uses cuML)
modal deploy modal_app.py

# Deploy hybrid (cost optimized)
modal deploy modal_gpu.py  # GPU worker first
modal deploy modal_web.py  # Then CPU web server
```

Please read the plan file and continue with remaining tasks (pre-seed layer caches, benchmark).

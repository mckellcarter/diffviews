# Continuation Prompt: DiffViews Hybrid Architecture

Paste this after clearing context to resume work.

---

## Prompt

I'm working on the diffviews visualizer with hybrid CPU/GPU Modal deployment. The full plan is in `docs/refactor-and-cuml-plan.md`.

### Current State (Deployed to Main)

**Hybrid Architecture:**
- `modal_web.py` — CPU container (Gradio UI, 30 min scaledown)
- `modal_gpu.py` — Lightweight GPU worker (torch only, 5-10s cold start)
- Mask computed on CPU from cached activations, sent to GPU for generation

**Completed Milestones:**
- M3: Modal compute (merged PR #76)
- M4: Cost optimization (T4 GPU, single-model loading)
- M5: App refactoring (60% reduction in app.py)
- M6: cuML integration (auto-detect backend)
- M7: Hybrid CPU/GPU (deployed, working)

### Architecture

```
┌─────────────────────────────────────────┐
│  CPU Container (modal_web.py)           │
│  - Gradio UI + all visualization        │
│  - compute_mask_dict() from activations │
│  - scaledown_window=1800s (30 min)      │
└──────────────────┬──────────────────────┘
                   │ Modal remote call
                   ▼
┌─────────────────────────────────────────┐
│  GPU Container (modal_gpu.py)           │
│  - Lightweight: torch, numpy, pillow    │
│  - generate_from_mask() only            │
│  - Cold start: 5-10s                    │
└─────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `modal_web.py` | CPU web server (Gradio UI) |
| `modal_gpu.py` | Lightweight GPU worker |
| `diffviews/visualization/gpu_ops.py` | Hybrid dispatch (set_remote_gpu_worker) |
| `diffviews/core/masking.py` | compute_mask_dict() for CPU-side mask |
| `scripts/check_layer_cache.py` | Check R2 layer cache status |

### Remaining Tasks

- [ ] Pre-seed all layer caches to R2 with cuML embeddings
- [ ] Benchmark cost savings (hybrid vs monolithic)
- [ ] Investigate multi-tab session corruption (documented in README Known Issues)

### Deployment

```bash
# Hybrid (recommended)
modal deploy modal_gpu.py  # GPU worker first
modal deploy modal_web.py  # CPU web server

# Force rebuild (pin commit hash in modal_web.py)
git rev-parse HEAD | head -c 7  # get hash
# Edit modal_web.py: @main → @<hash>
modal deploy modal_web.py
# Revert to @main after deploy
```

### Known Issues

**Multi-tab session corruption:** When multiple tabs hit the same Modal deployment, one tab's Gradio session can become stale. Workaround: switch to a different model and back. See README Known Issues.

### Commands

```bash
# Run tests
python -m pytest tests/ -v

# Check layer cache on R2
python scripts/check_layer_cache.py

# Local dev
diffviews viz --data-dir data
```

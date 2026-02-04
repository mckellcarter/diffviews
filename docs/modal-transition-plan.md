# DiffViews: HF Spaces → Modal Transition Plan

**Branch:** `feature/modal-transition` (from `main`)
**Status:** M1 code complete, HF rebuild in progress for E2E test

## Milestones

### M1: Cloudflare R2 Data Cache Layer ← CURRENT
Cache activations + UMAP layer embeddings on CF R2. HF remains compute host.

### M2: Model + Data Hosting on CF
Move checkpoints and base data to R2. HF no longer source of truth for data.

### M3: Compute Migration to Modal
Replace HF ZeroGPU with Modal serverless GPU. Full stack on CF (data) + Modal (compute).

---

## M1: CF R2 Layer Cache

### Architecture

```
layer selected
  → local disk hit? → load, done
  → R2 has it? → download csv/json/npy → refit reducer locally → done
  → full miss → GPU extract → UMAP → save local → async push to R2
```

### Key Decision: No PKL on R2
UMAP pickles contain numba JIT refs that break across environments. R2 stores only portable artifacts:
- `.csv` — UMAP coordinates + metadata
- `.json` — UMAP params
- `.npy` — raw flattened activations

Reducer is refit locally from cached activations+coordinates using `UMAP(init=coords, n_epochs=0)` (~2-5s vs ~30s extraction+UMAP).

### R2 Bucket

- **Bucket:** `diffviews` (created)
- **Region:** Auto
- **Public access:** Disabled
- **Key prefix structure:** flat keys, no pre-created dirs needed

```
data/{model}/layer_cache/{layer_name}.csv
data/{model}/layer_cache/{layer_name}.json
data/{model}/layer_cache/{layer_name}.npy
```

Cache self-populates as users select layers. Optional: pre-seed by running all layers once on HF after deploy.

### Implementation Status

| Step | File | Status |
|------|------|--------|
| R2 client module | `diffviews/data/r2_cache.py` | done |
| R2 client tests (16 tests) | `tests/test_r2_cache.py` | done |
| Add boto3 dep | `requirements.txt` | done |
| Wire into layer cache load + R2 fallback + pkl-less refit | `diffviews/visualization/app.py` `_load_layer_cache()` | done |
| Async upload after extraction | `diffviews/visualization/app.py` `recompute_layer_umap()` | done |
| Set HF Spaces Secrets | R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME | done |
| Update requirements.txt commit hash | pinned to `aef063b` (then `b1e62e7`) | done |
| HF rebuild + E2E test | deploy, select layer, verify R2 upload, delete local, re-select | **in progress** |

### Credentials (HF Spaces Secrets)
```
R2_ACCOUNT_ID=<cloudflare account id>
R2_ACCESS_KEY_ID=<r2 api token key id>
R2_SECRET_ACCESS_KEY=<r2 api token secret>
R2_BUCKET_NAME=diffviews
```

### Fallback
All R2 calls wrapped in try/except, return False on failure. `R2LayerCache.enabled=False` if env vars missing or boto3 unavailable. Full local-only flow preserved.

---

## M1 Completed Work

### PCA Pre-Reduction for UMAP Speed

**Problem:** UMAP on 1168 samples x 49K features takes 30s-5min on HF CPU.

**Solution:** Optional PCA reduction (49K → 50 dims) before UMAP. Controlled via env var.

| File | Change |
|------|--------|
| `diffviews/processing/umap.py` | `compute_umap()` gains `pca_components` param. PCA after StandardScaler, before UMAP. Returns 4-tuple: `(embeddings, reducer, scaler, pca_reducer)`. `save_embeddings()` persists PCA in pkl. |
| `app.py` | `get_pca_components()` reads `DIFFVIEWS_PCA_COMPONENTS` env (default `"50"`, `"0"`/`"none"` to disable). `regenerate_umap()` passes PCA through. |
| `diffviews/visualization/app.py` | `ModelData.umap_pca` field + backup/restore. `_load_layer_cache` loads PCA from pkl. `recompute_layer_umap` passes PCA config. Trajectory projection applies PCA before UMAP transform. |
| `tests/test_gradio_visualizer.py` | Fixed mock target bug (`extract_layer_activations` → `_extract_layer_on_gpu`), updated 4-tuple return. |

**Pipeline:** raw activations → StandardScaler → PCA(50) → UMAP(15, 0.1)

**Config:**
- `DIFFVIEWS_PCA_COMPONENTS=50` (default)
- `DIFFVIEWS_PCA_COMPONENTS=0` or `none` to disable

### R2LayerCache Module

`diffviews/data/r2_cache.py` — self-contained S3-compatible client:
- `layer_exists()` — HEAD check on .csv key
- `download_layer()` — downloads csv/json/npy to local dir
- `upload_layer()` — uploads csv/json/npy from local dir
- `upload_layer_async()` — daemon thread fire-and-forget upload
- Graceful degradation: `enabled=False` if creds missing or boto3 unavailable

### Layer Cache Load (pkl-less refit path)

`_load_layer_cache()` now supports three paths:
1. **Local pkl exists** → load reducer/scaler/pca from pkl (fast, existing behavior)
2. **R2 hit, no pkl** → download csv/npy → refit reducer from cached coords via `UMAP(init=coords, n_epochs=0)` → save pkl locally
3. **CSV only, no npy** → load embeddings for display, reducer stays None (no trajectory projection)

### Test Cleanup
- Removed `tests/test_visualizer.py` — obsolete Dash-era tests
- 143 tests passing (was 123)

---

## M2: CF Data Hosting (Planned)

Move all data from HF Hub to CF R2:
```
data/{model}/checkpoints/{model}.pkl        (~1GB each)
data/{model}/activations/imagenet_real/      (~50MB each)
data/{model}/embeddings/demo_embeddings.*    (~500MB each with pkl)
data/{model}/images/imagenet_real/           (~100MB each)
data/{model}/metadata/imagenet_real/         (<1MB)
```

### Changes Required
- Replace `ensure_data_ready()` HF `snapshot_download` with R2 bulk download
- Add `R2LayerCache.download_model_data()` method for full model data
- Keep HF as fallback during transition
- Consider chunked/streaming download for large checkpoints

---

## M3: Modal Compute Migration (Planned)

### Why Modal
- Serverless GPU, scales to zero (no idle cost)
- Full Docker control (install cuML, pin env)
- Better for large model support (TransformerLens integration)

### UMAP Strategy per Host

| Host | Strategy |
|------|----------|
| HF (M1-M2) | PCA(50) → CPU UMAP |
| Modal (M3) | cuML GPU UMAP (or PCA → UMAP if fast enough) |

### Key Changes for M3
- Replace `@spaces.GPU` with Modal `@app.function(gpu="A10G")`
- Remove ZeroGPU workarounds (codefind, module-level injection, picklability constraints)
- `app.py` becomes Modal entry point instead of HF Spaces entry
- Data loaded from R2 (M2) instead of HF Hub

---

## Auth & Privacy (Future, Post-M3)

- Keep public repo for M1-M2 (no secrets in code, R2 creds in env)
- Private fork or gitignored `.modal/` config when auth added
- SQL on CF only needed if user-generated data (saved workspaces, history)
- Flat files on R2 sufficient for activation/UMAP data

---

## UMAP Portability Notes

UMAP pickles are not portable across numba/Python versions due to JIT compilation. Options explored:

1. **Parametric UMAP** — neural net reducer, portable but adds TF/PyTorch dep
2. **Surrogate regressor** — sklearn MLP approximating the projection, fully portable
3. **n_epochs=0 refit** — pass cached coords as `init`, reconstruct reducer locally (~2-5s) ← **used in M1**
4. **Host-tagged pkls** — `{layer}.hf.pkl`, `{layer}.modal.pkl` per environment
5. **cuML GPU UMAP** — for Modal (M3), different library entirely

Current approach: portable artifacts on R2 + local refit. Revisit if refit latency is noticeable.

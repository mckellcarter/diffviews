# DiffViews: HF Spaces → Modal Transition Plan

**Branch:** `feature/modal-transition` (from `main`)
**Status:** M2 complete + E2E verified on HF, ready for PR

## Milestones

### M1: Cloudflare R2 Data Cache Layer ✓
Cache activations + UMAP layer embeddings on CF R2. HF remains compute host.

### M2: Model + Data Hosting on CF ✓
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

Reducer is refit locally via full `compute_umap()` from cached activations. `n_epochs=0`/`1` refit left `.transform()` internals degenerate — full fit required for trajectory projection.

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
| HF rebuild + E2E test | deploy, select layer, verify R2 upload, delete local, re-select | done |

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

### Layer Cache Load

`_load_layer_cache()` flow:
1. **Local cache miss** → try R2 download (csv/json/npy)
2. **Has activations (.npy)** → full `compute_umap()` fit → save pkl + updated CSV locally
3. **CSV only, no npy** → load embeddings for display, reducer stays None (no trajectory projection)

PKL is a local-only acceleration artifact — always overwritten by fresh fit from activations.

### Test Cleanup
- Removed `tests/test_visualizer.py` — obsolete Dash-era tests
- 143 tests passing (was 123)

---

## M2: CF Data Hosting ← CURRENT

Move all data from HF Hub to CF R2. R2-first download with HF fallback.

### What's on R2
```
data/{model}/config.json
data/{model}/checkpoints/*.pkl              (~1GB each)
data/{model}/activations/imagenet_real/*.npy + *.npy.json
data/{model}/embeddings/demo_embeddings.csv + .json  (no .pkl — numba)
data/{model}/metadata/imagenet_real/dataset_info.json
data/{model}/images/imagenet_real/sample_*.png
data/imagenet_standard_class_index.json
data/imagenet64_class_labels.json
```

**Skipped:** intermediates/, .npz (old), .pkl in embeddings/, layer_cache/

### Implementation Status

| Step | File | Status |
|------|------|--------|
| R2DataStore class | `diffviews/data/r2_cache.py` | done |
| R2DataStore tests (15 tests) | `tests/test_r2_cache.py` | done |
| Seeding script | `scripts/seed_r2.py` | done |
| Replace download_data() (R2 first, HF fallback) | `app.py` | done |
| Replace download_checkpoint() (R2 first, URL fallback) | `app.py` | done |
| Update CLI download_command() + --source flag | `diffviews/scripts/cli.py` | done |
| Seed R2 bucket | run `scripts/seed_r2.py --execute` | done (2362 files, 3.31GB) |
| E2E test on HF | verify R2 download on fresh start | done |
| M2 bug fixes | concurrent downloads, path fix, OOM, trajectory projection | done |

### Architecture
- `_make_r2_client()` shared helper for boto3 setup (used by R2DataStore + R2LayerCache)
- `R2DataStore.download_model_data()` downloads all files for a model
- `R2DataStore.download_prefix()` lists + downloads preserving key structure
- `download_data()` tries R2 first, falls back to HF `snapshot_download`
- `download_checkpoint()` tries R2 first, falls back to direct URL
- CLI `--source auto|r2|hf` flag for explicit control

### M2 Bug Fixes (E2E)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Slow R2 downloads (~1 img/sec) | `download_prefix()` sequential | `ThreadPoolExecutor(max_workers=8)` |
| 50GB disk exceeded | `download_model_data()` included `layer_cache/` | `exclude_dirs={"layer_cache"}` + LRU eviction (`DIFFVIEWS_LAYER_CACHE_MAX_MB`) |
| "no embeddings found" | `download_prefix` path bug — files landed outside model subdir | Target `local_dir / model` |
| OOM on layer switch (16GB limit) | Old 3GB+ activations held while loading new | `_clear_layer_data()` before load + `mmap_mode="r"` for .npy |
| Combo layer ValueError after switch | `get_default_layer_label()` read `umap_params` (overwritten) | Read from `default_umap_params` (immutable backup) |
| Trajectory divide-by-zero after layer switch | `n_epochs=0`/`1` refit left `.transform()` degenerate | Full `compute_umap()` fit from cached activations |

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
3. ~~**n_epochs=0 refit**~~ — degenerate `.transform()`, abandoned
4. **Full refit from cached activations** — `compute_umap()` on .npy, saves pkl locally ← **current approach**
5. **Host-tagged pkls** — `{layer}.hf.pkl`, `{layer}.modal.pkl` per environment
6. **cuML GPU UMAP** — for Modal (M3), different library entirely

Current approach: portable artifacts (csv/json/npy) on R2 + full local UMAP fit from activations. PKL is local-only cache, never uploaded to R2.

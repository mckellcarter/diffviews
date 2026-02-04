# Continuation Prompt: Modal Transition (M1→M3)

Paste this after clearing context to resume implementation.

---

## Prompt

I'm migrating the diffviews visualizer from HF Spaces to Modal with Cloudflare R2 for data hosting. The full plan is in `docs/modal-transition-plan.md` — read it first.

### What's Done (M1)

**PCA pre-reduction:** `compute_umap()` in `diffviews/processing/umap.py` now accepts `pca_components` param. Pipeline: StandardScaler → PCA(50) → UMAP. Returns 4-tuple: `(embeddings, reducer, scaler, pca_reducer)`. Controlled by `DIFFVIEWS_PCA_COMPONENTS` env var (default `"50"`, `"0"` to disable). `save_embeddings()` persists PCA reducer in pkl. All callers updated.

**R2 layer cache:** `diffviews/data/r2_cache.py` has `R2LayerCache` class (boto3 S3-compat). Methods: `layer_exists()`, `download_layer()`, `upload_layer()`, `upload_layer_async()`. Graceful degradation when creds missing. R2 stores only portable artifacts (csv/json/npy, no pkl — numba JIT breaks across envs).

**Integration in `diffviews/visualization/app.py`:**
- `GradioVisualizer.__init__` instantiates `R2LayerCache`
- `_load_layer_cache()`: local disk → R2 fallback → pkl-less refit via `UMAP(init=coords, n_epochs=0)` if only csv/npy from R2
- `recompute_layer_umap()`: after local save, calls `r2_cache.upload_layer_async()` (daemon thread, non-blocking)
- `ModelData` has `umap_pca` + `default_umap_pca` fields, trajectory projection applies PCA before UMAP transform

**R2 bucket:** `diffviews` created on Cloudflare. Keys: `data/{model}/layer_cache/{layer_name}.{csv,json,npy}`. HF Spaces Secrets configured (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME).

**Requirements:** `requirements.txt` pins `git+...@b1e62e7`, includes `boto3>=1.28.0`.

**Tests:** 143 passing. `tests/test_r2_cache.py` (16 tests, mocked boto3). Removed obsolete `tests/test_visualizer.py`.

**Status:** HF rebuild in progress for E2E test.

### What's Next

**M1 remaining:** E2E test on HF — verify R2 upload on layer selection, verify R2 download on cache miss.

**M2 (CF data hosting):** Move checkpoints (~1GB each), base activations (~50MB), images (~100MB), metadata from HF Hub to R2. Replace `ensure_data_ready()` HF `snapshot_download` with R2 bulk download. Key paths: `data/{model}/checkpoints/`, `data/{model}/activations/`, `data/{model}/images/`, `data/{model}/embeddings/`.

**M3 (Modal compute):** Replace `@spaces.GPU` with Modal `@app.function(gpu="A10G")`. Remove ZeroGPU workarounds (codefind, module-level injection, picklability). Consider cuML GPU UMAP. Data from R2 (M2).

### Key Architecture Notes

- Root `app.py` is HF Spaces entry; `@spaces.GPU` functions must live there (ZeroGPU codefind)
- GPU functions injected into submodule: `viz_mod._generate_on_gpu = generate_on_gpu`
- `_app_visualizer` module global for non-picklable objects (ZeroGPU forks subprocess)
- `regenerate_umap()` runs every HF startup for numba pkl compat
- Adapters: `DMD2ImageNetAdapter`, `EDMImageNetAdapter` in `diffviews/adapters/`
- Data: `data/{model}/` with config.json, checkpoints/, embeddings/, activations/, images/, metadata/

### Branch

`feature/modal-transition` (from `main`)

Please read the plan file and continue from where we left off.

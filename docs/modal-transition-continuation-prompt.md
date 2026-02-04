# Continuation Prompt: Modal Transition (M1→M3)

Paste this after clearing context to resume implementation.

---

## Prompt

I'm migrating the diffviews visualizer from HF Spaces to Modal with Cloudflare R2 for data hosting. The full plan is in `docs/modal-transition-plan.md` — read it first.

### What's Done

**M1 (R2 layer cache):** Complete, merged to main (PR #66). PCA pre-reduction, R2LayerCache for layer embeddings, pkl-less refit path, E2E verified on HF.

**M2 (CF data hosting) — code complete + R2 seeded, pending E2E on HF:**

**R2DataStore:** `diffviews/data/r2_cache.py` now has `R2DataStore` class alongside `R2LayerCache`. Shared `_make_r2_client()` helper. Methods: `list_objects()`, `file_exists()`, `download_file()`, `download_prefix()`, `download_model_data()`. Same graceful degradation pattern.

**R2-first downloads in `app.py`:**
- `download_data()` → tries `R2DataStore.download_model_data()` for each model, falls back to HF `snapshot_download`
- `download_checkpoint()` → tries R2 `download_file()` for checkpoint key, falls back to direct URL
- `ensure_data_ready()` unchanged (calls updated download functions)

**CLI update:** `diffviews/scripts/cli.py` `download_command()` uses R2-first with `--source auto|r2|hf` flag.

**Seeding script:** `scripts/seed_r2.py` walks local `data/`, uploads to R2. Skips intermediates/, .npz, .pkl in embeddings/, layer_cache/. Dry-run by default, `--execute` to upload. R2 seeded: 2362 files, 3.31GB, 0 failures.

**Tests:** 158 passing. 35 in `test_r2_cache.py` (16 R2LayerCache + 15 R2DataStore + 4 shared).

### What's Next

**M2 remaining:**
1. E2E test on HF: rebuild with updated commit hash, verify R2 download on fresh start
2. PR to main

**M3 (Modal compute):** Replace `@spaces.GPU` with Modal `@app.function(gpu="A10G")`. Remove ZeroGPU workarounds. Data from R2 (M2).

### Key Architecture Notes

- Root `app.py` is HF Spaces entry; `@spaces.GPU` functions must live there (ZeroGPU codefind)
- `_make_r2_client()` in `diffviews/data/r2_cache.py` shared by R2DataStore + R2LayerCache
- `regenerate_umap()` runs every HF startup — rebuilds .pkl from csv+activations (since .pkl not on R2)
- `download_prefix()` strips R2 key prefix to reconstruct local dir structure

### Branch

`feature/modal-transition` (from `main`)

Please read the plan file and continue from where we left off.

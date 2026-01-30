# Plan: On-Demand Layer Activation Extraction + UMAP

## Goal
Add layer selection dropdown to visualizer. When user picks a layer, extract activations for all 1168 samples through that layer, compute fresh UMAP, update plot. Cache results to disk.

## Key Insight
Current 1168 samples are **real ImageNet images** fed through the model at their `conditioning_sigma`. The forward pass is:
```python
img_tensor = (uint8_image / 127.5) - 1.0  # normalize to [-1, 1]
x_noisy = img_tensor * conditioning_sigma   # scale by sigma
output = adapter.forward(x_noisy, sigma_tensor, one_hot_labels)
# hooks capture intermediate activations during this forward pass
```
Images are already saved as PNGs in `{data_dir}/images/imagenet_real/`. Metadata has `image_path`, `conditioning_sigma`, `class_label` per sample.

## Files to Modify

| File | What |
|------|------|
| `diffviews/visualization/app.py` | All backend methods + UI changes |

Reuse existing APIs from `extractor.py` (`ActivationExtractor`), `umap.py` (`compute_umap`, `save_embeddings`). No changes needed to those files.

## Implementation Steps

### 1. Add `extract_layer_activations()` method to `GradioVisualizer`
- Takes `model_name`, `layer_name`, optional `progress_callback`
- Loads adapter, creates `ActivationExtractor(adapter, [layer_name])`
- Batched forward passes (batch_size=32) over all samples in `metadata_df`:
  - Load images from `image_path` PNGs -> uint8 (H,W,3) -> CHW tensor -> `(pixel / 127.5) - 1.0`
  - Scale: `x_noisy = img_tensor * conditioning_sigma`
  - Build one-hot class labels from `class_label`
  - Run `adapter.forward(x_noisy, sigma_tensor, one_hot_labels)` with hook registered
  - Capture activation from extractor, flatten (B, C, H, W) -> (B, C\*H\*W)
- Returns `(N, D)` numpy matrix
- Runs under `_generation_lock` (GPU access)

### 2. Add `recompute_layer_umap()` method to `GradioVisualizer`
- Takes `model_name`, `layer_name`, optional `progress_callback`
- **Check disk cache first**: `{data_dir}/embeddings/layer_cache/{layer_name}.csv/.pkl/.npy`
  - If cached, load via `_load_layer_cache()`, return early
- Call `extract_layer_activations()` (GPU, locked)
- Call `compute_umap()` from `processing/umap.py` (CPU, no lock needed)
- Fit new `NearestNeighbors` on embeddings
- **Save to disk cache** using `save_embeddings()` + `np.save()`
- **Atomic swap** on ModelData: update `df`, `activations`, `umap_reducer`, `umap_scaler`, `umap_params`, `nn_model`

### 3. Add `_load_layer_cache()` helper
- Load cached `.csv`, `.pkl`, `.npy` from `layer_cache/` dir
- Fit KNN, swap into ModelData

### 4. Add `_restore_default_embeddings()` helper
- Reload original pre-computed embeddings from `config["embeddings_path"]`
- Restore original activations, UMAP reducer/scaler, KNN
- Essentially re-runs the embedding-loading portion of `_load_model_data()`

### 5. Add `get_layer_choices()` method
- Returns `["default"] + adapter.hookable_layers`
- Requires adapter loaded; returns just `["default"]` if not yet loaded

### 6. UI: Add layer dropdown in left sidebar
- Between model selector row and status text
- Populated with `get_layer_choices()` (initially just "default", expands after adapter loads)
- Add `layer_status` Markdown for progress feedback
- CSS for `#layer-row` (same inline style as model-row)

### 7. Add `on_layer_change()` event handler
- If "default" selected: call `_restore_default_embeddings()`
- Otherwise: call `recompute_layer_umap()` with `gr.Progress()` callback
- Rebuild UMAP figure, reset selection state (selected_idx, neighbors, trajectories cleared)
- Update status text with layer name + sample count

### 8. Update `on_model_switch()`
- After switching model, load adapter (to populate hookable_layers)
- Reset layer dropdown to "default" with updated choices
- Add `layer_dropdown` to outputs

### 9. Generation consistency
- No structural changes needed to `on_generate` - it reads dynamically from `model_data.umap_params["layers"]`, `model_data.activations`, `model_data.umap_reducer`
- After layer change these are already updated, so trajectory projection + activation masking just work

## State Flow

```
Layer dropdown change
  -> "default": restore pre-computed data
  -> "encoder_block_0" etc:
      -> check disk cache -> load if exists
      -> else: load images from disk -> batched forward pass (GPU)
             -> compute UMAP (CPU) -> cache to disk
  -> swap ModelData fields atomically
  -> rebuild plot, reset selection
```

## Caching Strategy
- Disk path: `{model_data_dir}/embeddings/layer_cache/{layer_name}.csv` + `.json` + `.pkl` + `.npy`
- Uses existing `save_embeddings()` for CSV/JSON/PKL
- Separate `.npy` for raw activations (needed for generation masking)
- Cache key is layer name string (future: sorted concatenation for multi-layer)

## Performance Estimate
- 1168 samples @ batch_size=32 = ~37 batches, ~2s on GPU
- UMAP on (1168, ~49K features) = ~10-30s
- Total: ~30s with progress bar. Cached loads: <1s

## Future: Multi-Layer Extension
- Cache key becomes `"layer_a+layer_b"` (sorted, joined)
- Layer dropdown changes to multi-select with "Combine" toggle
- `extract_layer_activations` accepts list, concatenates
- No architectural changes needed - just UI and key generation

## Verification
1. Start app with existing data dir - should load default embeddings as before
2. Select a different layer (e.g. `encoder_block_0`) - should show progress, recompute, display new UMAP
3. Switch back to "default" - should restore original embeddings
4. Re-select same layer - should load from disk cache (fast)
5. Generate from neighbors after layer change - trajectory projection should use correct layer
6. Switch models - layer dropdown should reset to "default" with new model's layers

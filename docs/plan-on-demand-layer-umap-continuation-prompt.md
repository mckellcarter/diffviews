# Continuation Prompt: On-Demand Layer UMAP

Paste this after clearing context to resume implementation.

---

## Prompt

I'm implementing on-demand layer activation extraction + UMAP recomputation for the diffusion activation visualizer. The plan is in `docs/plan-on-demand-layer-umap.md` — read it first.

Key context:

**Architecture**: `diffviews/visualization/app.py` has `GradioVisualizer` class + `create_gradio_app()`. `ModelData` dataclass holds per-model state (df, activations, umap_reducer, umap_scaler, nn_model, adapter). Adapters are in `diffviews/adapters/` with `hookable_layers` property and `register_activation_hooks()`. Extraction uses `ActivationExtractor` from `diffviews/core/extractor.py`. UMAP uses `compute_umap()` and `save_embeddings()` from `diffviews/processing/umap.py`.

**Forward pass for extraction**: Real ImageNet images (64x64 PNGs) stored at `{data_dir}/images/imagenet_real/`. Each sample has `image_path`, `conditioning_sigma`, `class_label` in metadata. The forward pass is:
```python
img_tensor = (uint8_image / 127.5) - 1.0   # [-1, 1]
x_noisy = img_tensor * conditioning_sigma    # scale by sigma
adapter.forward(x_noisy, sigma_tensor, one_hot_labels)  # hooks capture activations
```

**Data source module**: `diffviews/data/sources.py` has `ImageNetDataSource` abstraction with `JPEGDataSource`, `NPZDataSource`, `LMDBDataSource` for loading images. Images are uint8 (B, 3, 64, 64) CHW format.

**Thread safety**: `_generation_lock` protects GPU access. ModelData fields need atomic swap after recomputation.

**UI**: Gradio 6 app. Left sidebar has model dropdown, preview, class filter, selected sample. Right sidebar has generation controls, neighbor gallery. Center is Plotly UMAP plot with JS bridge for click/hover. New layer dropdown goes between model selector and status text in left sidebar.

**What exists**: Pre-computed activations for `encoder_bottleneck+midblock` (2 layers concatenated, 98304 features). Stored as `.npy` (1168, 98304). Pre-computed UMAP embeddings as `.csv` + `.pkl` (reducer+scaler). These load at startup and should remain the "default" option.

**Branch**: `feature/noise-mode` (current working branch)

Please implement the plan step by step with manual edit approval. Start by reading the plan file and `app.py`, then proceed through the implementation steps.

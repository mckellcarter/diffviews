# DiffViews Project Notes

## HF Spaces / ZeroGPU Architecture

- Root `app.py` is the HF Spaces entry point (`app_file` in SPACES_README.md)
- `@spaces.GPU` decorators must be in root `app.py` — ZeroGPU's `codefind` only scans that file
- Setup runs at module level: `demo = _setup()` — Gradio hot-reload imports but never calls `main()`
- GPU functions injected into submodule: `viz_mod._generate_on_gpu = generate_on_gpu`
- Visualizer accessed via `_app_visualizer` module global, not function args (ZeroGPU forks subprocess, args must be picklable)
- `requirements.txt` pins `git+https://...@<exact_commit_hash>` — must update hash after merging submodule changes; `@main` gets cached by pip

## Key Files

- `app.py` — HF Spaces entry, `@spaces.GPU` functions, module-level setup
- `diffviews/visualization/app.py` — GradioVisualizer, create_gradio_app, all callbacks
- `diffviews/core/generator.py` — generate_with_mask_multistep
- `diffviews/core/extractor.py` — ActivationExtractor
- `diffviews/processing/umap.py` — compute_umap, save_embeddings
- `docs/plan-on-demand-layer-umap.md` — on-demand layer UMAP implementation plan

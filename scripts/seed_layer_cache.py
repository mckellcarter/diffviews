#!/usr/bin/env python3
"""Seed layer caches to R2 using Modal GPU.

Checks which layers are missing from R2 and seeds them using
the existing GradioVisualizer.recompute_layer_umap() method.

Usage:
    modal run scripts/seed_layer_cache.py
    modal run scripts/seed_layer_cache.py --model dmd2
    modal run scripts/seed_layer_cache.py --model dmd2 --layer encoder_bottleneck
    modal run scripts/seed_layer_cache.py --dry-run  # check only
"""

import modal

app = modal.App("diffviews-seed-cache")

# GPU image with cuML for fast UMAP
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "boto3>=1.28.0",
        "tqdm>=4.60.0",
        "plotly>=5.18.0",
        "gradio>=6.0.0",
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install("cuml-cu12>=25.02", "cupy-cuda12x>=12.0")
    .pip_install("umap-learn>=0.5.0")
    .pip_install("diffviews @ git+https://github.com/mckellcarter/diffviews.git@3dc6603")
)

vol = modal.Volume.from_name("diffviews-data", create_if_missing=True)
r2_secret = modal.Secret.from_name("R2_ACCESS")


@app.function(
    image=gpu_image,
    gpu="T4",
    volumes={"/data": vol},
    secrets=[r2_secret],
    timeout=3600,
)
def seed_layers(model_filter: str = None, layer_filter: str = None, dry_run: bool = False):
    """Seed missing layer caches to R2."""
    from pathlib import Path

    from diffviews.data.r2_cache import R2DataStore, R2LayerCache
    from diffviews.visualization.visualizer import GradioVisualizer
    from diffviews.visualization.gpu_ops import set_visualizer

    DATA_DIR = Path("/data")

    # Download data from R2 if needed
    print("Checking/downloading data from R2...")
    store = R2DataStore()
    if not store.enabled:
        print("ERROR: R2 not configured")
        return {"error": "R2 not configured"}

    for model in ["dmd2", "edm"]:
        config = DATA_DIR / model / "config.json"
        if not config.exists():
            print(f"Downloading {model} data...")
            store.download_model_data(model, DATA_DIR)

    # Initialize visualizer (loads adapters, has recompute_layer_umap)
    print("\nInitializing visualizer...")
    visualizer = GradioVisualizer(data_dir=DATA_DIR, device="cuda")
    set_visualizer(visualizer)  # Required for _extract_layer_on_gpu

    # Get models to process
    models = list(visualizer.model_configs.keys())
    if model_filter:
        models = [m for m in models if m == model_filter]

    print(f"Models: {models}")

    r2_cache = R2LayerCache()
    results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Ensure model is loaded
        if not visualizer._ensure_model_loaded(model_name):
            print(f"  Failed to load model")
            continue

        model_data = visualizer.get_model(model_name)
        if model_data is None:
            continue

        # Get hookable layers from adapter
        adapter = visualizer.load_adapter(model_name)
        if adapter is None:
            print(f"  No adapter (no checkpoint?)")
            continue

        hookable_layers = adapter.hookable_layers
        print(f"  Hookable layers: {hookable_layers}")

        # Filter layers if specified
        layers_to_check = hookable_layers
        if layer_filter:
            layers_to_check = [l for l in hookable_layers if l == layer_filter]

        # Check which are fully cached on R2 (csv + npy required)
        def layer_complete(model: str, layer: str) -> bool:
            """Check if layer has all required files on R2."""
            for ext in [".csv", ".json", ".npy", ".pkl"]:
                key = f"data/{model}/layer_cache/{layer}{ext}"
                try:
                    r2_cache._client.head_object(Bucket=r2_cache._bucket, Key=key)
                except Exception:
                    return False
            return True

        cached = []
        missing = []
        for layer in layers_to_check:
            if layer_complete(model_name, layer):
                cached.append(layer)
            else:
                missing.append(layer)

        print(f"  Cached: {cached}")
        print(f"  Missing: {missing}")

        if dry_run:
            results[model_name] = {"cached": cached, "missing": missing, "action": "dry_run"}
            continue

        # Seed missing layers with retry until complete
        seeded = []
        failed = []
        cache_dir = model_data.data_dir / "embeddings" / "layer_cache"
        max_retries = 3

        for layer_name in missing:
            print(f"\n  Seeding {layer_name}...")

            for attempt in range(max_retries):
                # Check if already complete on R2 (from previous attempt)
                if layer_complete(model_name, layer_name):
                    print(f"  ✓ {layer_name} verified complete on R2")
                    seeded.append(layer_name)
                    break

                if attempt > 0:
                    print(f"  Retry {attempt + 1}/{max_retries}...")

                # Clear any incomplete local cache
                local_files = [cache_dir / f"{layer_name}{ext}" for ext in [".csv", ".json", ".npy", ".pkl"]]
                existing = [f for f in local_files if f.exists()]
                if existing:
                    print(f"  Clearing local cache ({len(existing)} files)...")
                    for f in existing:
                        f.unlink()

                # Direct extraction (bypass recompute_layer_umap which may load partial R2 cache)
                print(f"  Extracting activations...")
                activations = visualizer.extract_layer_activations(model_name, layer_name)
                if activations is None:
                    print(f"  ✗ Extraction failed")
                    continue

                # Compute UMAP
                print(f"  Computing UMAP...")
                from diffviews.processing.umap import compute_umap, save_embeddings
                embeddings, reducer, scaler, pca_reducer = compute_umap(
                    activations, n_neighbors=15, min_dist=0.1, normalize=True, pca_components=50
                )

                # Build df with UMAP coords
                import pandas as pd
                new_df = pd.read_csv(model_data.data_dir / "metadata" / "activation_metadata.csv")
                new_df["umap_x"] = embeddings[:, 0]
                new_df["umap_y"] = embeddings[:, 1]
                umap_params = {"layers": [layer_name], "n_neighbors": 15, "min_dist": 0.1}

                # Save all files
                cache_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np
                csv_path = cache_dir / f"{layer_name}.csv"
                save_embeddings(embeddings, new_df, csv_path, umap_params, reducer, scaler, pca_reducer)
                np.save(cache_dir / f"{layer_name}.npy", activations)
                print(f"  Saved to {cache_dir}")

                # Verify all local files exist before upload
                local_complete = all(f.exists() for f in local_files)
                if not local_complete:
                    missing_local = [f.name for f in local_files if not f.exists()]
                    print(f"  ⚠ Missing local files: {missing_local}")
                    continue

                # Sync upload all files
                print(f"  Uploading to R2 (sync)...")
                r2_cache.upload_layer(model_name, layer_name, cache_dir)

                # Verify upload complete
                if layer_complete(model_name, layer_name):
                    print(f"  ✓ {layer_name} seeded and verified")
                    seeded.append(layer_name)
                    break
                else:
                    print(f"  ⚠ Upload incomplete, will retry...")
            else:
                print(f"  ✗ {layer_name} failed after {max_retries} attempts")
                failed.append(layer_name)

        vol.commit()
        results[model_name] = {"cached": cached, "seeded": seeded, "failed": failed}

    print(f"\n{'='*60}")
    print("Summary:")
    for model, info in results.items():
        line = f"  {model}: cached={len(info.get('cached', []))}, seeded={len(info.get('seeded', []))}"
        if info.get('failed'):
            line += f", FAILED={info['failed']}"
        print(line)

    return results


@app.local_entrypoint()
def main(model: str = None, layer: str = None, dry_run: bool = False):
    """CLI entrypoint."""
    print(f"Seeding layer caches (model={model}, layer={layer}, dry_run={dry_run})")
    result = seed_layers.remote(model_filter=model, layer_filter=layer, dry_run=dry_run)
    print(f"\nResult: {result}")

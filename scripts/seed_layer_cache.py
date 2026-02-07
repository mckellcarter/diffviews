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
    .pip_install("diffviews @ git+https://github.com/mckellcarter/diffviews.git@main")
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

        # Check which are cached on R2
        cached = []
        missing = []
        for layer in layers_to_check:
            if r2_cache.layer_exists(model_name, layer):
                cached.append(layer)
            else:
                missing.append(layer)

        print(f"  Cached: {cached}")
        print(f"  Missing: {missing}")

        if dry_run:
            results[model_name] = {"cached": cached, "missing": missing, "action": "dry_run"}
            continue

        # Seed missing layers using existing method
        seeded = []
        for layer_name in missing:
            print(f"\n  Seeding {layer_name}...")
            success = visualizer.recompute_layer_umap(model_name, layer_name)
            if success:
                print(f"  ✓ {layer_name} seeded")
                seeded.append(layer_name)
            else:
                print(f"  ✗ {layer_name} failed")

        vol.commit()
        results[model_name] = {"cached": cached, "seeded": seeded}

    print(f"\n{'='*60}")
    print("Summary:")
    for model, info in results.items():
        print(f"  {model}: cached={info.get('cached', [])}, seeded={info.get('seeded', info.get('missing', []))}")

    return results


@app.local_entrypoint()
def main(model: str = None, layer: str = None, dry_run: bool = False):
    """CLI entrypoint."""
    print(f"Seeding layer caches (model={model}, layer={layer}, dry_run={dry_run})")
    result = seed_layers.remote(model_filter=model, layer_filter=layer, dry_run=dry_run)
    print(f"\nResult: {result}")

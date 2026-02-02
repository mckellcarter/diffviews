"""
HuggingFace Spaces entry point for diffviews.

This file is the main entry point for HF Spaces deployment.
It downloads required data and checkpoints on startup, then launches the Gradio app.

Requirements:
    Python 3.10+
    Gradio 6.0+

Environment variables:
    DIFFVIEWS_DATA_DIR: Override data directory (default: data)
    DIFFVIEWS_CHECKPOINT: Which checkpoint to download (dmd2, edm, all, none; default: all)
    DIFFVIEWS_DEVICE: Override device (cuda, mps, cpu; auto-detected if not set)
"""

import os
from pathlib import Path

import spaces

# Data source configuration
DATA_REPO_ID = "mckell/diffviews_demo_data"
CHECKPOINT_URLS = {
    "dmd2": (
        "https://huggingface.co/mckell/diffviews-dmd2-checkpoint/"
        "resolve/main/dmd2-imagenet-64-10step.pkl"
    ),
    "edm": (
        "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
        "edm-imagenet-64x64-cond-adm.pkl"
    ),
}
CHECKPOINT_FILENAMES = {
    "dmd2": "dmd2-imagenet-64-10step.pkl",
    "edm": "edm-imagenet-64x64-cond-adm.pkl",
}


def download_data(output_dir: Path) -> None:
    """Download data from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading data from {DATA_REPO_ID}...")
    print(f"Output directory: {output_dir.absolute()}")

    snapshot_download(
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        local_dir=output_dir,
        revision="main",
    )
    print(f"Data downloaded to {output_dir}")


def download_checkpoint(output_dir: Path, model: str) -> None:
    """Download model checkpoint."""
    import urllib.request

    if model not in CHECKPOINT_URLS:
        print(f"Unknown model: {model}")
        return

    ckpt_dir = output_dir / model / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filename = CHECKPOINT_FILENAMES[model]
    filepath = ckpt_dir / filename

    if filepath.exists():
        print(f"Checkpoint exists: {filepath}")
        return

    url = CHECKPOINT_URLS[model]
    print(f"Downloading {model} checkpoint (~1GB)...")
    print(f"  URL: {url}")

    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  Done ({filepath.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  Error downloading checkpoint: {e}")
        print("  Generation will be disabled without checkpoint")


def regenerate_umap(data_dir: Path, model: str) -> bool:
    """Regenerate UMAP pickle for a model to ensure numba compatibility.

    This recomputes UMAP from activations and saves new pickle file.
    Required when running on different environment than original pickle was created.
    """
    from diffviews.processing.umap import (
        load_dataset_activations,
        compute_umap,
        save_embeddings,
    )
    import json

    model_dir = data_dir / model
    activation_dir = model_dir / "activations" / "imagenet_real"
    metadata_path = model_dir / "metadata" / "imagenet_real" / "dataset_info.json"
    embeddings_dir = model_dir / "embeddings"

    # Check if activations exist
    if not activation_dir.exists() or not metadata_path.exists():
        print(f"  Skipping UMAP regeneration for {model}: missing activations")
        return False

    # Find existing embeddings CSV to get parameters
    csv_files = list(embeddings_dir.glob("*.csv"))
    if not csv_files:
        print(f"  Skipping UMAP regeneration for {model}: no embeddings CSV")
        return False

    csv_path = csv_files[0]
    json_path = csv_path.with_suffix(".json")
    pkl_path = csv_path.with_suffix(".pkl")

    # Load UMAP params from existing JSON
    umap_params = {"n_neighbors": 15, "min_dist": 0.1, "layers": ["encoder_bottleneck", "midblock"]}
    if json_path.exists():
        with open(json_path, "r") as f:
            umap_params = json.load(f)

    print(f"  Regenerating UMAP for {model}...")
    print(f"    Params: n_neighbors={umap_params.get('n_neighbors', 15)}, min_dist={umap_params.get('min_dist', 0.1)}")

    try:
        # Load activations
        activations, metadata_df = load_dataset_activations(activation_dir, metadata_path)
        print(f"    Loaded {activations.shape[0]} activations")

        # Compute UMAP
        embeddings, reducer, scaler = compute_umap(
            activations,
            n_neighbors=umap_params.get("n_neighbors", 15),
            min_dist=umap_params.get("min_dist", 0.1),
            normalize=True,
        )

        # Save (overwrites existing pickle with compatible version)
        save_embeddings(embeddings, metadata_df, csv_path, umap_params, reducer, scaler)
        print(f"    UMAP pickle regenerated: {pkl_path}")
        return True

    except Exception as e:
        print(f"    Error regenerating UMAP: {e}")
        return False


def check_umap_compatibility(data_dir: Path, model: str) -> bool:
    """Check if UMAP pickle is compatible with current numba environment."""
    embeddings_dir = data_dir / model / "embeddings"
    pkl_files = list(embeddings_dir.glob("*.pkl"))

    if not pkl_files:
        return True  # No pickle to check

    pkl_path = pkl_files[0]

    try:
        import pickle
        with open(pkl_path, "rb") as f:
            umap_data = pickle.load(f)

        reducer = umap_data.get("reducer")
        if reducer is None:
            return True

        # Try a dummy transform to check numba compatibility
        import numpy as np
        dummy = np.random.randn(1, 100).astype(np.float32)

        # This will fail if numba JIT is incompatible
        scaler = umap_data.get("scaler")
        if scaler:
            dummy_scaled = scaler.transform(dummy)
        else:
            dummy_scaled = dummy

        # The actual transform - this triggers numba JIT
        _ = reducer.transform(dummy_scaled)
        return True

    except Exception as e:
        print(f"  UMAP compatibility check failed for {model}: {e}")
        return False


def ensure_data_ready(data_dir: Path, checkpoints: list) -> bool:
    """Ensure data and checkpoints are downloaded."""
    print(f"Checking for existing data in {data_dir.absolute()}...")

    # Check which models have data (config + embeddings + images)
    models_with_data = []
    for model in ["dmd2", "edm"]:
        config_path = data_dir / model / "config.json"
        embeddings_dir = data_dir / model / "embeddings"
        images_dir = data_dir / model / "images" / "imagenet_real"

        if not config_path.exists():
            continue
        if not embeddings_dir.exists():
            continue

        csv_files = list(embeddings_dir.glob("*.csv"))
        png_files = list(images_dir.glob("sample_*.png")) if images_dir.exists() else []

        if csv_files and png_files:
            models_with_data.append(model)
            print(f"  Found {model}: {len(csv_files)} csv, {len(png_files)} images")

    if not models_with_data:
        print("Data not found, downloading...")
        download_data(data_dir)
    else:
        print(f"Data already present: {models_with_data}")

    # Download checkpoints only if not present
    for model in checkpoints:
        download_checkpoint(data_dir, model)

    # Regenerate UMAP for all models to ensure numba compatibility
    # This is fast enough to do on every startup and avoids compatibility issues
    print("\nRegenerating UMAP pickles for numba compatibility...")
    for model in ["dmd2", "edm"]:
        model_dir = data_dir / model
        if not model_dir.exists():
            print(f"  {model}: model dir not found, skipping")
            continue

        embeddings_dir = model_dir / "embeddings"
        if not embeddings_dir.exists() or not list(embeddings_dir.glob("*.csv")):
            print(f"  {model}: no embeddings found, skipping")
            continue

        print(f"  {model}: regenerating UMAP...")
        regenerate_umap(data_dir, model)

    return True


def get_device() -> str:
    """Auto-detect best available device."""
    override = os.environ.get("DIFFVIEWS_DEVICE")
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@spaces.GPU(duration=120)
def generate_on_gpu(
    visualizer, model_name, all_neighbors, class_label,
    n_steps, m_steps, s_max, s_min, guidance, noise_mode,
    extract_layers, can_project
):
    """Run masked generation on GPU. Must live in app_file for ZeroGPU detection."""
    from diffviews.core.masking import ActivationMasker
    from diffviews.core.generator import generate_with_mask_multistep

    with visualizer._generation_lock:
        adapter = visualizer.load_adapter(model_name)
        if adapter is None:
            return None

        activation_dict = visualizer.prepare_activation_dict(model_name, all_neighbors)
        if activation_dict is None:
            return None

        masker = ActivationMasker(adapter)
        for layer_name, activation in activation_dict.items():
            masker.set_mask(layer_name, activation)
        masker.register_hooks(list(activation_dict.keys()))

        try:
            result = generate_with_mask_multistep(
                adapter,
                masker,
                class_label=class_label,
                num_steps=int(n_steps),
                mask_steps=int(m_steps),
                sigma_max=float(s_max),
                sigma_min=float(s_min),
                guidance_scale=float(guidance),
                noise_mode=(noise_mode or "stochastic noise").replace(" noise", ""),
                num_samples=1,
                device=visualizer.device,
                extract_layers=extract_layers if can_project else None,
                return_trajectory=can_project,
                return_intermediates=True,
                return_noised_inputs=True,
            )
        finally:
            masker.remove_hooks()

    return result


@spaces.GPU(duration=180)
def extract_layer_on_gpu(visualizer, model_name, layer_name, batch_size=32):
    """Extract layer activations on GPU. Must live in app_file for ZeroGPU detection."""
    return visualizer.extract_layer_activations(model_name, layer_name, batch_size)


def main():
    """Main entry point for HF Spaces."""
    # Configuration from environment
    data_dir = Path(os.environ.get("DIFFVIEWS_DATA_DIR", "data"))
    checkpoint_config = os.environ.get("DIFFVIEWS_CHECKPOINT", "all")  # Download all by default
    device = get_device()

    # Parse checkpoint config
    if checkpoint_config == "all":
        checkpoints = list(CHECKPOINT_URLS.keys())
    elif checkpoint_config == "none":
        checkpoints = []
    else:
        checkpoints = [c.strip() for c in checkpoint_config.split(",") if c.strip()]

    print("=" * 50)
    print("DiffViews - Diffusion Activation Visualizer")
    print("=" * 50)
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Device: {device}")
    print(f"Checkpoints: {checkpoints}")
    print("=" * 50)

    # Ensure data is ready
    ensure_data_ready(data_dir, checkpoints)

    # Import and launch visualizer
    import gradio as gr
    from diffviews.visualization.app import (
        GradioVisualizer,
        create_gradio_app,
        CUSTOM_CSS,
        PLOTLY_HANDLER_JS,
    )

    # Inject ZeroGPU-decorated functions into visualization module
    # so Gradio callbacks use the versions codefind can detect
    import diffviews.visualization.app as viz_mod
    viz_mod._generate_on_gpu = generate_on_gpu
    viz_mod._extract_layer_on_gpu = extract_layer_on_gpu

    print("\nInitializing visualizer...")
    visualizer = GradioVisualizer(
        data_dir=data_dir,
        device=device,
    )

    print("Creating Gradio app...")
    app = create_gradio_app(visualizer)

    print("Launching...")
    # HF Spaces expects server on 0.0.0.0:7860
    app.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Spaces handles public URL
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        js=PLOTLY_HANDLER_JS,
    )


if __name__ == "__main__":
    main()

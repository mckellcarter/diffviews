"""
HuggingFace Spaces entry point for diffviews.

This file is the main entry point for HF Spaces deployment.
It downloads required data and checkpoints on startup, then launches the Gradio app.

Environment variables:
    DIFFVIEWS_DATA_DIR: Override data directory (default: data)
    DIFFVIEWS_CHECKPOINT: Which checkpoint to download (dmd2, edm, all, none; default: dmd2)
    DIFFVIEWS_DEVICE: Override device (cuda, mps, cpu; auto-detected if not set)
"""

import os
from pathlib import Path

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


def main():
    """Main entry point for HF Spaces."""
    # Configuration from environment
    data_dir = Path(os.environ.get("DIFFVIEWS_DATA_DIR", "data"))
    checkpoint_config = os.environ.get("DIFFVIEWS_CHECKPOINT", "dmd2")
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
    from diffviews.visualization.app import GradioVisualizer, create_gradio_app

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
        show_api=False,  # Disable API docs to avoid schema generation bug
    )


if __name__ == "__main__":
    main()

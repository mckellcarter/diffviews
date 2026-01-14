#!/usr/bin/env python
"""Download model data from HuggingFace Hub."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DATA_REPO_ID = "mckell/diffviews_demo_data"
CHECKPOINT_URLS = {
    "dmd2": "https://huggingface.co/mckell/diffviews-dmd2-checkpoint/resolve/main/dmd2-imagenet-64-10step.pkl",
    "edm": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
}
CHECKPOINT_FILENAMES = {
    "dmd2": "dmd2-imagenet-64-10step.pkl",
    "edm": "edm-imagenet-64x64-cond-adm.pkl",
}


def download_data(output_dir: str = "data", revision: str = "main"):
    """Download all model data (config, images, embeddings, activations)."""
    output_path = Path(output_dir)

    print(f"Downloading data from {DATA_REPO_ID}...")
    print(f"Output directory: {output_path.absolute()}")

    snapshot_download(
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        local_dir=output_path,
        revision=revision,
    )

    print(f"\nData downloaded to {output_path}")
    return output_path


def download_checkpoint(model: str, output_dir: str = "data"):
    """Download checkpoint for a specific model."""
    if model not in CHECKPOINT_URLS:
        print(f"Unknown model: {model}. Available: {list(CHECKPOINT_URLS.keys())}")
        return None

    import urllib.request

    output_path = Path(output_dir) / model / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)

    filename = CHECKPOINT_FILENAMES[model]
    filepath = output_path / filename

    if filepath.exists():
        print(f"Checkpoint already exists: {filepath}")
        return filepath

    url = CHECKPOINT_URLS[model]
    print(f"Downloading {model} checkpoint...")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    urllib.request.urlretrieve(url, filepath)
    print(f"  Done ({filepath.stat().st_size / 1e6:.1f} MB)")

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Download model data and checkpoints from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        choices=["dmd2", "edm", "all", "none"],
        default=["all"],
        help="Checkpoints to download (default: all)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision for data repo",
    )
    args = parser.parse_args()

    # Download data
    download_data(args.output_dir, args.revision)

    # Download checkpoints
    checkpoints = args.checkpoints
    if "all" in checkpoints:
        checkpoints = list(CHECKPOINT_URLS.keys())
    elif "none" in checkpoints:
        checkpoints = []

    for model in checkpoints:
        download_checkpoint(model, args.output_dir)

    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print(f"\nRun the visualizer:")
    print(f"  python -m diffviews.visualization.app --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()

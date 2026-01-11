#!/usr/bin/env python
"""Download DMD2 checkpoint from HuggingFace Hub."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ID = "mckell/diffviews-dmd2-checkpoint"
DEFAULT_OUTPUT = "checkpoints/dmd2-imagenet-10step"


def download_checkpoint(output_dir: str = DEFAULT_OUTPUT, revision: str = "main"):
    """Download checkpoint files from HuggingFace Hub."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint from {REPO_ID}...")
    print(f"Output directory: {output_path.absolute()}")

    # Download all files in repo
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=output_path,
        revision=revision,
    )

    print(f"\nCheckpoint downloaded to {output_path}")
    print(f"\nTo use with the visualizer:")
    print(f"  python -m diffviews.visualization.app \\")
    print(f"    --checkpoint_path {output_path} \\")
    print(f"    --adapter dmd2-imagenet-64 \\")
    print(f"    --data_dir demo_data \\")
    print(f"    --embeddings demo_data/embeddings/demo_embeddings.csv")


def main():
    parser = argparse.ArgumentParser(description="Download DMD2 checkpoint")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision to download",
    )
    args = parser.parse_args()

    download_checkpoint(args.output_dir, args.revision)


if __name__ == "__main__":
    main()

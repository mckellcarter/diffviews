"""
CLI for diffviews package.

Usage:
    diffviews download [--output-dir DIR] [--checkpoints all|dmd2|edm|none]
    diffviews convert <data_dir> [--model-type TYPE] [--keep-npz]
    diffviews visualize [--data-dir DIR] [--port PORT] [--debug] [...]
"""

import argparse
import sys
from pathlib import Path

DATA_REPO_ID = "mckell/diffviews_demo_data"
CHECKPOINT_URLS = {
    "dmd2": "https://huggingface.co/mckell/diffviews-dmd2-checkpoint/resolve/main/dmd2-imagenet-64-10step.pkl",
    "edm": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
}
CHECKPOINT_FILENAMES = {
    "dmd2": "dmd2-imagenet-64-10step.pkl",
    "edm": "edm-imagenet-64x64-cond-adm.pkl",
}


def download_command(args):
    """Download data and checkpoints from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("  pip install huggingface_hub")
        sys.exit(1)

    import urllib.request

    output_path = Path(args.output_dir)

    # Download data
    print(f"Downloading data from {DATA_REPO_ID}...")
    print(f"Output directory: {output_path.absolute()}")

    snapshot_download(
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        local_dir=output_path,
        revision="main",
    )
    print(f"Data downloaded to {output_path}")

    # Download checkpoints
    checkpoints = args.checkpoints
    if "all" in checkpoints:
        checkpoints = list(CHECKPOINT_URLS.keys())
    elif "none" in checkpoints:
        checkpoints = []

    for model in checkpoints:
        if model not in CHECKPOINT_URLS:
            print(f"Unknown model: {model}")
            continue

        ckpt_dir = output_path / model / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        filename = CHECKPOINT_FILENAMES[model]
        filepath = ckpt_dir / filename

        if filepath.exists():
            print(f"Checkpoint exists: {filepath}")
            continue

        url = CHECKPOINT_URLS[model]
        print(f"Downloading {model} checkpoint...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Done ({filepath.stat().st_size / 1e6:.1f} MB)")

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nRun the visualizer:")
    print(f"  diffviews viz --data-dir {args.output_dir}")


def convert_command(args):
    """Convert .npz activations to fast .npy format."""
    from ..core.extractor import convert_to_fast_format

    data_dir = Path(args.data_dir)
    activation_dir = data_dir / "activations" / args.model_type

    if not activation_dir.exists():
        print(f"Error: Activation directory not found: {activation_dir}")
        sys.exit(1)

    npz_files = list(activation_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {activation_dir}")
        sys.exit(0)

    print(f"Converting {len(npz_files)} .npz files in {activation_dir}")

    for npz_path in npz_files:
        npy_path = npz_path.with_suffix('.npy')
        if npy_path.exists():
            print(f"  Skipping {npz_path.name} (.npy exists)")
            continue

        print(f"  Converting {npz_path.name}...")
        convert_to_fast_format(npz_path)

        if not args.keep_npz:
            npz_path.unlink()
            print(f"    Removed {npz_path.name}")

    print("Done!")


def visualize_command(args):
    """Launch the visualization app."""
    from ..utils.device import get_device
    from ..visualization.app import DMD2Visualizer

    visualizer = DMD2Visualizer(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings,
        checkpoint_path=args.checkpoint_path,
        device=get_device(args.device),
        num_steps=args.num_steps,
        mask_steps=args.mask_steps,
        guidance_scale=args.guidance_scale,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        label_dropout=args.label_dropout,
        adapter_name=args.adapter,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        max_classes=args.max_classes,
        initial_model=args.model
    )

    visualizer.run(debug=args.debug, port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="diffviews",
        description="Diffusion activation visualizer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download subcommand
    download_parser = subparsers.add_parser(
        "download",
        help="Download demo data and checkpoints from HuggingFace"
    )
    download_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)"
    )
    download_parser.add_argument(
        "--checkpoints",
        nargs="*",
        choices=["dmd2", "edm", "all", "none"],
        default=["all"],
        help="Checkpoints to download (default: all)"
    )

    # Convert subcommand
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert .npz activations to fast .npy format (~30x speedup)"
    )
    convert_parser.add_argument(
        "data_dir",
        type=str,
        help="Data directory (e.g., data/dmd2)"
    )
    convert_parser.add_argument(
        "--model-type",
        type=str,
        default="imagenet_real",
        help="Model type subdirectory (default: imagenet_real)"
    )
    convert_parser.add_argument(
        "--keep-npz",
        action="store_true",
        help="Keep original .npz files after conversion"
    )

    # Visualize subcommand
    viz_parser = subparsers.add_parser(
        "visualize",
        aliases=["viz"],
        help="Launch the visualization app"
    )
    viz_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory"
    )
    viz_parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to precomputed embeddings CSV"
    )
    viz_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run server on"
    )
    viz_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    viz_parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint for generation"
    )
    viz_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device for generation (auto-detected if not specified)"
    )
    viz_parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of denoising steps"
    )
    viz_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="CFG scale"
    )
    viz_parser.add_argument(
        "--sigma-max",
        type=float,
        default=80.0,
        help="Maximum sigma for denoising"
    )
    viz_parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.5,
        help="Minimum sigma for denoising"
    )
    viz_parser.add_argument(
        "--label-dropout",
        type=float,
        default=0.0,
        help="Label dropout for CFG models"
    )
    viz_parser.add_argument(
        "--mask-steps",
        type=int,
        default=1,
        help="Steps to apply activation mask"
    )
    viz_parser.add_argument(
        "--adapter",
        type=str,
        default="dmd2-imagenet-64",
        help="Adapter name for model loading"
    )
    viz_parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter"
    )
    viz_parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter"
    )
    viz_parser.add_argument(
        "--max-classes", "-c",
        type=int,
        default=None,
        help="Maximum classes to load"
    )
    viz_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Initial model to load (e.g., dmd2, edm)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "download":
        download_command(args)
    elif args.command == "convert":
        convert_command(args)
    elif args.command in ("visualize", "viz"):
        visualize_command(args)


if __name__ == "__main__":
    main()

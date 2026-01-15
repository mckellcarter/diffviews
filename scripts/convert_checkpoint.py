#!/usr/bin/env python
"""Convert DMD2 checkpoint from safetensors to pickle format.

This creates a self-contained pickle file that can be loaded without
the DMD2 model architecture code (like EDM checkpoints).
"""

import argparse
import pickle
import sys
from pathlib import Path

# Add DMD2 to path for model imports
DMD2_PATH = Path(__file__).parent.parent.parent / 'DMD2'
if DMD2_PATH.exists():
    sys.path.insert(0, str(DMD2_PATH))


def convert_checkpoint(input_dir: str, output_path: str, device: str = 'cpu'):
    """Convert safetensors checkpoint to pickle format."""
    import torch
    from safetensors.torch import load_file

    # Import model architecture from DMD2
    from third_party.edm.training.networks import EDMPrecond
    from main.edm.edm_network import get_imagenet_edm_config

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    print(f"Loading checkpoint from {input_dir}")

    # Build model with config
    config = {
        "img_resolution": 64,
        "img_channels": 3,
        "label_dim": 1000,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet"
    }
    config.update(get_imagenet_edm_config(label_dropout=0.0))

    model = EDMPrecond(**config)

    # Remove augment mapping (not needed for inference)
    if hasattr(model.model, 'map_augment'):
        del model.model.map_augment
        model.model.map_augment = None

    # Load weights from safetensors
    safetensors_path = input_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {input_dir}")

    print(f"Loading weights from {safetensors_path}")
    state_dict = load_file(safetensors_path, device=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    # Save as pickle (same format as EDM)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}")
    data = {'ema': model}
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Converted checkpoint saved to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert DMD2 checkpoint to pickle")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input checkpoint directory (with model.safetensors)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading (default: cpu)",
    )
    args = parser.parse_args()

    convert_checkpoint(args.input_dir, args.output_path, args.device)


if __name__ == "__main__":
    main()

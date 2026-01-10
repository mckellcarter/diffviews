"""
Checkpoint loading utilities.
Handles .pth, .safetensors, and accelerator directories.
"""

import os
import torch


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load model state dict from various checkpoint formats.

    Args:
        checkpoint_path: Path to checkpoint file or directory. Supports:
            - .pth files (PyTorch standard)
            - .safetensors files
            - Directories containing model.safetensors or pytorch_model.bin
        device: Device to map tensors to (default: "cpu")

    Returns:
        state_dict: Model state dictionary

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist
    """
    checkpoint_path = str(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        # Accelerator checkpoint directory
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            print(f"Loaded checkpoint from {safetensors_path}")
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location=device)
            print(f"Loaded checkpoint from {pytorch_path}")
        else:
            raise FileNotFoundError(
                f"No model file found in directory {checkpoint_path}. "
                f"Expected model.safetensors or pytorch_model.bin"
            )
    elif checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Standard .pth checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return state_dict

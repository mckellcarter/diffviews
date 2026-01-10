"""
Device detection and management utilities.
Supports CUDA, MPS (Apple Silicon), and CPU.
"""

import torch


def get_device(prefer_device: str = None) -> str:
    """
    Get best available device.

    Args:
        prefer_device: Optional preferred device ('cuda', 'mps', 'cpu')

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if prefer_device:
        if prefer_device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        elif prefer_device == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        elif prefer_device == 'cpu':
            return 'cpu'
        else:
            print(f"Warning: {prefer_device} not available, falling back to auto-detection")

    # Auto-detect
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon) device")
    else:
        device = 'cpu'
        print("Using CPU device")

    return device


def get_device_info(device: str) -> dict:
    """Get information about the device."""
    info = {'device': device}

    if device == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3
        info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**3
    elif device == 'mps':
        info['device_name'] = 'Apple Silicon MPS'
        info['memory_allocated'] = 'N/A'
        info['memory_reserved'] = 'N/A'
    else:
        info['device_name'] = 'CPU'
        info['memory_allocated'] = 'N/A'
        info['memory_reserved'] = 'N/A'

    return info


def move_to_device(model, device: str):
    """
    Move model to device with proper handling.

    Args:
        model: PyTorch model
        device: Target device string

    Returns:
        Model on device
    """
    if device == 'mps':
        # MPS sometimes has issues with .to(device)
        try:
            return model.to(device).float()
        except Exception as e:
            print(f"Warning: MPS placement failed ({e}), falling back to CPU")
            return model.to('cpu').float()
    else:
        return model.to(device)

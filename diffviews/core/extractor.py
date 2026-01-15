"""
Activation extraction using adapter interface.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..adapters.base import GeneratorAdapter


class ActivationExtractor:
    """
    Extract activations from a model during forward pass.

    Uses the adapter interface for model-agnostic hook registration.
    """

    def __init__(self, adapter: GeneratorAdapter, layers: List[str] = None):
        """
        Args:
            adapter: GeneratorAdapter instance
            layers: Layer names to extract (default: adapter.hookable_layers[:2])
        """
        self.adapter = adapter
        self.layers = layers or adapter.hookable_layers[:2]
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, name: str):
        """Create forward hook that stores activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        """Register extraction hooks on specified layers."""
        for name in self.layers:
            hook_fn = self._make_hook(name)
            handles = self.adapter.register_activation_hooks([name], hook_fn)
            self._handles.extend(handles)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        """Clear stored activations."""
        self.activations.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get extracted activations."""
        return self.activations.copy()

    def save(self, output_path: Path, metadata: Dict = None):
        """
        Save activations to disk.

        Args:
            output_path: Path to save (without extension)
            metadata: Optional metadata dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten and save as numpy
        activation_dict = {}
        for name, activation in self.activations.items():
            if len(activation.shape) == 4:
                B, C, H, W = activation.shape
                activation_dict[name] = activation.reshape(B, -1).numpy()
            else:
                activation_dict[name] = activation.numpy()

        np.savez_compressed(
            str(output_path.with_suffix('.npz')),
            **activation_dict
        )

        if metadata:
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def flatten_activations(activations: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten all layer activations to single vector per sample.

    Args:
        activations: Dict of layer_name -> array (B, C*H*W)

    Returns:
        Flattened array (B, total_features)
    """
    all_features = []
    for layer_name in sorted(activations.keys()):
        act = activations[layer_name]
        if len(act.shape) > 2:
            act = act.reshape(act.shape[0], -1)
        all_features.append(act)

    return np.concatenate(all_features, axis=1)


def load_activations(activation_path: Path):
    """
    Load activations and metadata from disk.

    Returns:
        (activations_dict, metadata_dict)
    """
    activation_path = Path(activation_path)

    data = np.load(str(activation_path.with_suffix('.npz')))
    activations = {key: data[key] for key in data.keys()}

    metadata = {}
    metadata_path = activation_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return activations, metadata


def convert_to_fast_format(
    npz_path: Path,
    output_path: Path = None,
    layers: List[str] = None
) -> Path:
    """
    Convert compressed .npz to fast-loading .npy format.

    Concatenates specified layers into single pre-flattened array.

    Args:
        npz_path: Path to .npz file
        output_path: Output .npy path (default: same name with .npy)
        layers: Layer names to include (default: all, sorted)

    Returns:
        Path to created .npy file
    """
    npz_path = Path(npz_path)
    if output_path is None:
        output_path = npz_path.with_suffix('.npy')

    data = np.load(str(npz_path))
    layer_names = layers or sorted(data.keys())

    # Concatenate layers in sorted order
    arrays = [data[name] for name in layer_names]
    combined = np.concatenate(arrays, axis=1).astype(np.float32)

    np.save(str(output_path), combined)

    # Save layer info for reconstruction
    info_path = output_path.with_suffix('.npy.json')
    info = {
        'layers': layer_names,
        'shapes': {name: list(data[name].shape) for name in layer_names},
        'total_features': combined.shape[1],
        'num_samples': combined.shape[0]
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    return output_path


def load_fast_activations(npy_path: Path, mmap_mode: str = 'r') -> np.ndarray:
    """
    Load pre-concatenated activations with memory mapping.

    Args:
        npy_path: Path to .npy file
        mmap_mode: 'r' for read-only mmap, None for full load

    Returns:
        Activation matrix (N, D)
    """
    return np.load(str(npy_path), mmap_mode=mmap_mode)

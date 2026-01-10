"""
Activation extraction using adapter interface.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json

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

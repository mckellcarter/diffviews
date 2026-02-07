"""
Activation masking using adapter interface.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..adapters.base import GeneratorAdapter


class ActivationMasker:
    """
    Mask (replace) layer activations with fixed values during forward pass.

    Uses the adapter interface for model-agnostic hook registration.
    """

    def __init__(self, adapter: GeneratorAdapter):
        """
        Args:
            adapter: GeneratorAdapter instance
        """
        self.adapter = adapter
        self.masks: Dict[str, torch.Tensor] = {}
        self._handles = []

    def set_mask(self, layer_name: str, activation: torch.Tensor):
        """
        Set fixed activation for a layer.

        Args:
            layer_name: Layer to mask
            activation: Tensor to use as fixed output
        """
        self.masks[layer_name] = activation.cpu()

    def clear_mask(self, layer_name: str):
        """Remove mask for a layer."""
        self.masks.pop(layer_name, None)

    def clear_masks(self):
        """Remove all masks."""
        self.masks.clear()

    def _make_hook(self, name: str):
        """Create forward hook that replaces output with mask."""
        def hook(module, input, output):
            if name not in self.masks:
                return output

            mask = self.masks[name]

            if isinstance(output, tuple):
                target = output[0]
                masked = mask.to(target.device, target.dtype)
                if masked.shape[0] == 1 and target.shape[0] > 1:
                    masked = masked.expand(target.shape[0], -1, -1, -1)
                return (masked,) + output[1:]
            else:
                masked = mask.to(output.device, output.dtype)
                if masked.shape[0] == 1 and output.shape[0] > 1:
                    masked = masked.expand(output.shape[0], -1, -1, -1)
                return masked

        return hook

    def register_hooks(self, layers: List[str] = None):
        """
        Register masking hooks.

        Args:
            layers: Layers to mask (default: all layers with masks set)
        """
        if layers is None:
            layers = list(self.masks.keys())

        for name in layers:
            hook_fn = self._make_hook(name)
            handles = self.adapter.register_activation_hooks([name], hook_fn)
            self._handles.extend(handles)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def load_activation_from_npz(npz_path, layer_name: str) -> torch.Tensor:
    """
    Load activation from saved NPZ file.

    Args:
        npz_path: Path to .npz file
        layer_name: Layer to load

    Returns:
        Activation tensor (1, C*H*W)
    """
    data = np.load(npz_path)
    if layer_name not in data:
        available = list(data.keys())
        raise ValueError(f"Layer '{layer_name}' not found. Available: {available}")

    activation = torch.from_numpy(data[layer_name])
    if len(activation.shape) == 1:
        activation = activation.unsqueeze(0)
    return activation


def compute_mask_dict(
    activations: np.ndarray,
    neighbor_indices: List[int],
    layer_shapes: Dict[str, tuple],
    layers: List[str] = None,
) -> Dict[str, np.ndarray]:
    """Compute mask dict from cached activations (CPU, pure numpy).

    This is the CPU-side computation for hybrid mode. Returns numpy arrays
    that can be serialized and sent to GPU worker.

    Args:
        activations: (N, D) full activation matrix
        neighbor_indices: Indices to average
        layer_shapes: {layer_name: (C, H, W)}
        layers: Layers to include (default: all in layer_shapes, sorted)

    Returns:
        Dict mapping layer_name -> (1, C, H, W) numpy array
    """
    if len(neighbor_indices) == 0:
        return {}

    if layers is None:
        layers = sorted(layer_shapes.keys())

    # Average neighbor activations
    neighbor_acts = activations[neighbor_indices]  # (K, D)
    center_activation = np.mean(neighbor_acts, axis=0, keepdims=True)  # (1, D)

    # Split by layer (MUST be sorted order!)
    mask_dict = {}
    offset = 0
    for layer_name in sorted(layers):
        if layer_name not in layer_shapes:
            continue
        shape = layer_shapes[layer_name]  # (C, H, W)
        size = int(np.prod(shape))
        layer_act_flat = center_activation[0, offset:offset + size]
        layer_act = layer_act_flat.reshape(1, *shape)  # (1, C, H, W)
        mask_dict[layer_name] = layer_act.astype(np.float32)
        offset += size

    return mask_dict


def unflatten_activation(flat_activation: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Reshape flattened activation to spatial dimensions.

    Args:
        flat_activation: (1, C*H*W) or (C*H*W,) tensor
        target_shape: (C, H, W) original shape

    Returns:
        Reshaped tensor (1, C, H, W)
    """
    if len(flat_activation.shape) == 1:
        flat_activation = flat_activation.unsqueeze(0)

    B = flat_activation.shape[0]
    C, H, W = target_shape
    return flat_activation.reshape(B, C, H, W)

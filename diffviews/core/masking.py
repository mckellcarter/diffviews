"""
UMAP-specific mask computation for visualization.

Core masking (ActivationMasker, unflatten_activation) moved to adapt_diff.
"""

from typing import Dict, List

import numpy as np


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
        layer_act_flat = center_activation[0, offset : offset + size]
        layer_act = layer_act_flat.reshape(1, *shape)  # (1, C, H, W)
        mask_dict[layer_name] = layer_act.astype(np.float32)
        offset += size

    return mask_dict

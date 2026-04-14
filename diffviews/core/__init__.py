"""Core activation extraction and masking functionality."""

# Re-export from adapt_diff (authoritative source)
from adapt_diff import (
    ActivationExtractor,
    ActivationMasker,
    flatten_activations,
    load_activations,
    convert_to_fast_format,
    load_fast_activations,
    unflatten_activation,
    load_activation_from_npz,
)

# Keep UMAP-specific function local
from .masking import compute_mask_dict

__all__ = [
    # From adapt_diff
    "ActivationExtractor",
    "ActivationMasker",
    "flatten_activations",
    "load_activations",
    "convert_to_fast_format",
    "load_fast_activations",
    "unflatten_activation",
    "load_activation_from_npz",
    # Local
    "compute_mask_dict",
]

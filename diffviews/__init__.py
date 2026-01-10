"""
DiffViews: Diffusion model activation visualizer.

A model-agnostic visualization toolkit for exploring diffusion model
activations through UMAP embeddings and interactive generation.
"""

__version__ = "0.1.0"

from .adapters.base import GeneratorAdapter
from .adapters.registry import get_adapter, list_adapters, register_adapter

__all__ = [
    "GeneratorAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]

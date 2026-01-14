"""Model adapter interface and registry."""

from .base import GeneratorAdapter
from .hooks import HookMixin
from .registry import get_adapter, list_adapters, register_adapter, discover_adapters

# Import adapters to register them
from .dmd2_imagenet import DMD2ImageNetAdapter
from .edm_imagenet import EDMImageNetAdapter

__all__ = [
    "GeneratorAdapter",
    "HookMixin",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "discover_adapters",
    "DMD2ImageNetAdapter",
    "EDMImageNetAdapter",
]

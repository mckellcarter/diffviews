"""Model adapter interface and registry."""

from .base import GeneratorAdapter

# Import adapters to register them
from .dmd2_imagenet import DMD2ImageNetAdapter
from .edm_imagenet import EDMImageNetAdapter
from .hooks import HookMixin
from .registry import discover_adapters, get_adapter, list_adapters, register_adapter

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

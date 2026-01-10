"""Model adapter interface and registry."""

from .base import GeneratorAdapter
from .hooks import HookMixin
from .registry import get_adapter, list_adapters, register_adapter, discover_adapters

__all__ = [
    "GeneratorAdapter",
    "HookMixin",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "discover_adapters",
]

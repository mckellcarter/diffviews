"""Data loading and class label utilities."""

from .class_labels import load_class_labels, get_class_name
from .sources import NPZDataSource, JPEGDataSource

# LMDB is optional
try:
    from .lmdb_utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb
    from .sources import LMDBDataSource
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False

__all__ = [
    "load_class_labels",
    "get_class_name",
    "NPZDataSource",
    "JPEGDataSource",
    "LMDB_AVAILABLE",
]

if LMDB_AVAILABLE:
    __all__.extend(["LMDBDataSource", "retrieve_row_from_lmdb", "get_array_shape_from_lmdb"])

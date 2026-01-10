"""
ImageNet class label utilities.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np


# Module-level cache
_class_labels: Optional[Dict[int, str]] = None
_imagenet64_to_standard: Optional[Dict[int, int]] = None


def get_data_dir() -> Path:
    """Get bundled data directory (diffviews/data/)."""
    return Path(__file__).parent


def load_class_labels(labels_path: Optional[Path] = None) -> Dict[int, str]:
    """
    Load ImageNet class labels.

    Args:
        labels_path: Path to class labels JSON (default: bundled)

    Returns:
        Dict mapping class_id -> class_name
    """
    global _class_labels

    if _class_labels is not None:
        return _class_labels

    if labels_path is None:
        labels_path = get_data_dir() / "imagenet_standard_class_index.json"

    with open(labels_path, 'r') as f:
        raw_labels = json.load(f)

    # Format: {str_id: [synset, name]} -> {int_id: name}
    _class_labels = {int(k): v[1] for k, v in raw_labels.items()}
    return _class_labels


def get_class_name(class_id: int, labels_path: Optional[Path] = None) -> str:
    """Get class name for a class ID."""
    labels = load_class_labels(labels_path)
    return labels.get(class_id, f"unknown_{class_id}")


def load_class_labels_map(labels_path: Optional[Path] = None) -> Dict:
    """
    Load full class labels map with synsets.

    Returns:
        Dict of {str_id: [synset, name]}
    """
    if labels_path is None:
        labels_path = get_data_dir() / "imagenet_standard_class_index.json"

    with open(labels_path, 'r') as f:
        return json.load(f)


def load_imagenet64_to_standard_mapping(
    imagenet64_path: Optional[Path] = None,
    standard_path: Optional[Path] = None
) -> Dict[int, int]:
    """
    Load mapping from ImageNet64 indices to standard ImageNet indices.

    ImageNet64 uses a different class ordering than standard ImageNet-1K.
    """
    global _imagenet64_to_standard

    if _imagenet64_to_standard is not None:
        return _imagenet64_to_standard

    data_dir = get_data_dir()

    if imagenet64_path is None:
        imagenet64_path = data_dir / "imagenet64_class_labels.json"
    if standard_path is None:
        standard_path = data_dir / "imagenet_standard_class_index.json"

    if not imagenet64_path.exists():
        print(f"Warning: ImageNet64 labels not found at {imagenet64_path}")
        return {}

    # Load both label files
    with open(imagenet64_path, 'r') as f:
        imagenet64_labels = json.load(f)

    with open(standard_path, 'r') as f:
        standard_labels = json.load(f)

    # Build synset -> standard index mapping
    synset_to_standard = {v[0]: int(k) for k, v in standard_labels.items()}

    # Build ImageNet64 -> Standard mapping
    _imagenet64_to_standard = {}
    for idx64_str, (synset, name) in imagenet64_labels.items():
        idx64 = int(idx64_str)
        if synset in synset_to_standard:
            _imagenet64_to_standard[idx64] = synset_to_standard[synset]

    return _imagenet64_to_standard


def remap_imagenet64_labels_to_standard(labels: np.ndarray) -> np.ndarray:
    """
    Remap ImageNet64 labels to standard ImageNet indices.

    Args:
        labels: Array of ImageNet64 class indices (0-999)

    Returns:
        Array of standard ImageNet class indices (0-999)
    """
    mapping = load_imagenet64_to_standard_mapping()

    if not mapping:
        return labels

    remapped = np.zeros_like(labels)
    for idx64, idx_std in mapping.items():
        remapped[labels == idx64] = idx_std

    return remapped

"""
Data source abstractions for ImageNet activation extraction.
Supports LMDB, NPZ, and JPEG directory formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

from .class_labels import remap_imagenet64_labels_to_standard, load_class_labels_map


class ImageNetDataSource(ABC):
    """Abstract interface for ImageNet data loading."""

    @abstractmethod
    def scan_samples(
        self,
        target_classes: Optional[List[int]],
        samples_per_class: int,
        num_samples: int
    ) -> List[int]:
        """Return indices of samples matching criteria."""
        pass

    @abstractmethod
    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a batch of samples.

        Returns:
            (images, labels, source_paths) where:
                - images: uint8 array (B, 3, 64, 64)
                - labels: int64 array (B,) in source ordering
                - source_paths: List of source identifiers
        """
        pass

    @abstractmethod
    def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
        """Convert source labels to standard ImageNet-1K ordering."""
        pass

    @abstractmethod
    def close(self):
        """Release resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class NPZDataSource(ImageNetDataSource):
    """ImageNet64 NPZ batches - requires label remapping."""

    def __init__(self, npz_dir: Path):
        self.npz_dir = Path(npz_dir)
        self.npz_files = sorted(
            list(self.npz_dir.glob('*.npz')),
            key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 0
        )
        if not self.npz_files:
            raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

        # Build index mapping
        self.npz_batch_sizes = []
        self.idx_to_npz = []
        total = 0

        for file_idx, npz_file in enumerate(self.npz_files):
            data = np.load(npz_file)
            batch_size = data['data'].shape[0]
            self.npz_batch_sizes.append(batch_size)
            for within_idx in range(batch_size):
                self.idx_to_npz.append((file_idx, within_idx))
            total += batch_size

        self.total_samples = total
        print(f"Found {len(self.npz_files)} NPZ files with {self.total_samples:,} samples")

    def scan_samples(
        self,
        target_classes: Optional[List[int]],
        samples_per_class: int,
        num_samples: int
    ) -> List[int]:
        if target_classes is None:
            target_classes = list(range(1000))

        target_set = set(target_classes)
        class_counts = {c: 0 for c in target_classes}
        selected_indices = []
        global_idx = 0

        for npz_file in self.npz_files:
            if len(selected_indices) >= num_samples:
                break

            data = np.load(npz_file)
            labels_0indexed = data['labels'] - 1

            for label in labels_0indexed:
                if label in target_set and class_counts[label] < samples_per_class:
                    selected_indices.append(global_idx)
                    class_counts[label] += 1
                    if len(selected_indices) >= num_samples:
                        break
                global_idx += 1

        return selected_indices

    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        batch_size = len(indices)
        images = np.zeros((batch_size, 3, 64, 64), dtype=np.uint8)
        labels = np.zeros(batch_size, dtype=np.int64)
        paths = []

        # Group by NPZ file
        npz_groups: Dict[int, List[Tuple[int, int]]] = {}
        for local_idx, global_idx in enumerate(indices):
            file_idx, within_idx = self.idx_to_npz[global_idx]
            if file_idx not in npz_groups:
                npz_groups[file_idx] = []
            npz_groups[file_idx].append((local_idx, within_idx))

        for npz_idx, items in npz_groups.items():
            npz_file = self.npz_files[npz_idx]
            data = np.load(npz_file)
            images_flat = data['data']
            labels_1indexed = data['labels']

            for local_idx, file_within_idx in items:
                images[local_idx] = images_flat[file_within_idx].reshape(3, 64, 64)
                labels[local_idx] = labels_1indexed[file_within_idx] - 1

        for global_idx in indices:
            paths.append(f"npz_sample_{global_idx}")

        return images, labels, paths

    def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
        return remap_imagenet64_labels_to_standard(labels)

    def close(self):
        pass


class JPEGDataSource(ImageNetDataSource):
    """Directory of JPEG images organized by synset."""

    def __init__(self, imagenet_dir: Path, split: str = "train", class_labels_map: Optional[Dict] = None):
        self.imagenet_dir = Path(imagenet_dir)
        self.split = split
        self.split_dir = self.imagenet_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"ImageNet split not found: {self.split_dir}")

        if class_labels_map is None:
            class_labels_map = load_class_labels_map()

        # Build synset -> class_id mapping
        self.synset_to_class = {
            v[0]: int(k) for k, v in class_labels_map.items()
        }

        # Collect image paths
        extensions = ['.JPEG', '.jpg', '.png']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.split_dir.rglob(f'*{ext}')))

        self.total_samples = len(self.image_paths)
        print(f"Found {self.total_samples:,} images in {self.split_dir}")

    def scan_samples(
        self,
        target_classes: Optional[List[int]],
        samples_per_class: int,
        num_samples: int
    ) -> List[int]:
        indices = list(range(len(self.image_paths)))
        np.random.shuffle(indices)
        return indices[:num_samples]

    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        batch_size = len(indices)
        images = np.zeros((batch_size, 3, 64, 64), dtype=np.uint8)
        labels = np.zeros(batch_size, dtype=np.int64)
        paths = []

        for i, idx in enumerate(indices):
            img_path = self.image_paths[idx]

            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.uint8)
            images[i] = img_array.transpose(2, 0, 1)

            synset_id = img_path.parent.name
            labels[i] = self.synset_to_class.get(synset_id, -1)
            paths.append(str(img_path))

        return images, labels, paths

    def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
        return labels

    def close(self):
        pass


# LMDB support (optional)
try:
    import lmdb
    from .lmdb_utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb

    class LMDBDataSource(ImageNetDataSource):
        """LMDB dataset - labels already in standard ordering."""

        def __init__(self, lmdb_path: Path):
            self.lmdb_path = Path(lmdb_path)
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            self.image_shape = get_array_shape_from_lmdb(self.env, 'images')
            self.label_shape = get_array_shape_from_lmdb(self.env, 'labels')
            self.total_samples = self.image_shape[0]
            print(f"LMDB: {self.total_samples:,} samples, shape: {self.image_shape[1:]}")

        def scan_samples(
            self,
            target_classes: Optional[List[int]],
            samples_per_class: int,
            num_samples: int
        ) -> List[int]:
            if target_classes is None:
                target_classes = list(range(1000))

            target_set = set(target_classes)
            class_counts = {c: 0 for c in target_classes}
            selected_indices = []

            for idx in tqdm(range(self.total_samples), desc="Scanning"):
                label = retrieve_row_from_lmdb(
                    self.env, "labels", np.int64, self.label_shape[1:], idx
                )
                label_int = int(label.item()) if hasattr(label, 'item') else int(label)

                if label_int in target_set and class_counts[label_int] < samples_per_class:
                    selected_indices.append(idx)
                    class_counts[label_int] += 1
                    if len(selected_indices) >= num_samples:
                        break

            return selected_indices

        def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
            batch_size = len(indices)
            images = np.zeros((batch_size, 3, 64, 64), dtype=np.uint8)
            labels = np.zeros(batch_size, dtype=np.int64)
            paths = []

            for i, idx in enumerate(indices):
                images[i] = retrieve_row_from_lmdb(
                    self.env, "images", np.uint8, self.image_shape[1:], idx
                )
                label = retrieve_row_from_lmdb(
                    self.env, "labels", np.int64, self.label_shape[1:], idx
                )
                labels[i] = int(label.item()) if hasattr(label, 'item') else int(label)
                paths.append(f"lmdb_idx_{idx}")

            return images, labels, paths

        def get_standard_labels(self, labels: np.ndarray) -> np.ndarray:
            return labels

        def close(self):
            self.env.close()

except ImportError:
    # LMDB not available
    pass


def create_data_source(
    lmdb_path: Optional[Path] = None,
    npz_dir: Optional[Path] = None,
    imagenet_dir: Optional[Path] = None,
    split: str = "train",
    class_labels_map: Optional[Dict] = None
) -> ImageNetDataSource:
    """
    Factory to create appropriate data source.

    Args:
        lmdb_path: Path to LMDB dataset
        npz_dir: Directory containing NPZ batch files
        imagenet_dir: Root directory of ImageNet JPEG dataset
        split: Dataset split for JPEG format
        class_labels_map: Class labels map (required for JPEG)

    Returns:
        ImageNetDataSource implementation
    """
    if lmdb_path is not None:
        try:
            return LMDBDataSource(lmdb_path)
        except NameError:
            raise ImportError("LMDB support requires 'lmdb' package: pip install lmdb")
    if npz_dir is not None:
        return NPZDataSource(npz_dir)
    if imagenet_dir is not None:
        return JPEGDataSource(imagenet_dir, split, class_labels_map)
    raise ValueError("Must provide one of: lmdb_path, npz_dir, imagenet_dir")

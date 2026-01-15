"""
UMAP embedding computation for activation visualization.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from ..core.extractor import flatten_activations, load_activations, load_fast_activations


def load_dataset_activations(
    activation_dir: Path,
    metadata_path: Path,
    max_samples: Optional[int] = None,
    batch_size: int = 500,
    low_memory: bool = False,
    use_mmap: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load all activations from dataset.

    Prefers fast .npy format if available, falls back to .npz.

    Args:
        activation_dir: Directory containing activation files
        metadata_path: Path to dataset_info.json
        max_samples: Optional limit on samples
        batch_size: Samples per batch (only for .npz fallback)
        low_memory: Use memory-mapped temp file (only for .npz)
        use_mmap: Use memory mapping for .npy files (fast)

    Returns:
        (activation_matrix, metadata_df)
    """
    with open(metadata_path, 'r') as f:
        dataset_info = json.load(f)

    samples = dataset_info['samples'][:max_samples] if max_samples else dataset_info['samples']

    # Check for fast .npy format first
    npy_files = list(activation_dir.glob("*.npy"))
    if npy_files:
        return _load_fast_format(activation_dir, samples, npy_files[0], use_mmap)

    # Fallback to slow .npz loading
    return _load_npz_format(activation_dir, samples, batch_size, low_memory)


def _load_fast_format(
    activation_dir: Path,
    samples: list,
    npy_path: Path,
    use_mmap: bool
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load from pre-concatenated .npy file."""
    import time
    start = time.time()

    mmap_mode = 'r' if use_mmap else None
    activation_matrix = load_fast_activations(npy_path, mmap_mode=mmap_mode)

    # If mmap, copy to regular array for modification safety
    if use_mmap:
        activation_matrix = np.array(activation_matrix)

    elapsed = time.time() - start
    print(f"Fast-loaded {len(samples)} samples in {elapsed:.2f}s from {npy_path.name}")

    metadata_df = pd.DataFrame(samples)
    return activation_matrix, metadata_df


def _load_npz_format(
    activation_dir: Path,
    samples: list,
    batch_size: int,
    low_memory: bool
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load from compressed .npz files (slow fallback)."""
    import tempfile

    print(f"Loading {len(samples)} samples in batches of {batch_size} (slow .npz format)...")
    print("Tip: Run 'diffviews convert-activations' for 30x faster loading")

    # Determine activation shape from first sample
    first_sample = samples[0]

    if 'activation_path' in first_sample:
        data_root = activation_dir.parent.parent
        first_path = data_root / first_sample['activation_path']
    else:
        first_path = activation_dir / f"{first_sample['sample_id']}.npz"

    first_act, _ = load_activations(first_path)
    first_flat = flatten_activations(first_act)
    activation_dim = first_flat.shape[1]

    # Preallocate array
    if low_memory:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        activation_matrix = np.memmap(
            temp_file.name,
            dtype=np.float32,
            mode='w+',
            shape=(len(samples), activation_dim)
        )
    else:
        activation_matrix = np.zeros((len(samples), activation_dim), dtype=np.float32)

    metadata_records = []
    valid_idx = 0
    batch_cache = {}

    for i in tqdm(range(0, len(samples), batch_size), desc="Loading"):
        batch_samples = samples[i:i+batch_size]

        for sample in batch_samples:
            sample_id = sample['sample_id']

            if 'batch_index' in sample:
                act_path_str = sample['activation_path']
                batch_index = sample['batch_index']
                data_root = activation_dir.parent.parent
                act_path = data_root / act_path_str
            else:
                act_path = activation_dir / f"{sample_id}.npz"
                batch_index = 0

            if not act_path.with_suffix('.npz').exists():
                print(f"Warning: Missing {act_path.with_suffix('.npz')}")
                continue

            act_path_key = str(act_path)
            if act_path_key not in batch_cache:
                activations, _ = load_activations(act_path)
                batch_cache[act_path_key] = activations
            else:
                activations = batch_cache[act_path_key]

            flat_act = flatten_activations(activations)
            activation_matrix[valid_idx] = flat_act[batch_index]

            metadata_records.append(sample)
            valid_idx += 1

    # Trim and cleanup
    if low_memory:
        result = np.array(activation_matrix[:valid_idx])
        del activation_matrix
        import os
        os.unlink(temp_file.name)
        activation_matrix = result
    else:
        activation_matrix = activation_matrix[:valid_idx]

    metadata_df = pd.DataFrame(metadata_records)

    print(f"Loaded activations: {activation_matrix.shape}")

    # Handle NaN/inf
    nan_count = np.isnan(activation_matrix).sum()
    inf_count = np.isinf(activation_matrix).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaN and {inf_count} inf values, replacing with 0")
        activation_matrix = np.nan_to_num(activation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return activation_matrix, metadata_df


def compute_umap(
    activations: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    n_components: int = 2,
    random_state: Optional[int] = 42,
    normalize: bool = True
) -> Tuple[np.ndarray, UMAP, Optional[StandardScaler]]:
    """
    Compute UMAP projection.

    Args:
        activations: (N, D) activation matrix
        n_neighbors: UMAP n_neighbors
        min_dist: UMAP min_dist
        metric: Distance metric
        n_components: Output dimensions
        random_state: Random seed
        normalize: Whether to normalize before UMAP

    Returns:
        (embeddings, reducer, scaler)
    """
    print(f"\nComputing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")

    scaler = None
    if normalize:
        print("Normalizing activations...")
        scaler = StandardScaler()
        activations = scaler.fit_transform(activations)

    print("Running UMAP...")
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        verbose=True,
    )

    embeddings = reducer.fit_transform(activations)

    print(f"UMAP embeddings: {embeddings.shape}")
    return embeddings, reducer, scaler


def save_embeddings(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    output_path: Path,
    umap_params: dict,
    reducer: Optional[UMAP] = None,
    scaler: Optional[StandardScaler] = None
):
    """
    Save UMAP embeddings + metadata to CSV.

    Args:
        embeddings: (N, 2) or (N, 3) UMAP coordinates
        metadata_df: DataFrame with sample metadata
        output_path: Output CSV path
        umap_params: Dict of UMAP parameters
        reducer: Fitted UMAP model
        scaler: Fitted StandardScaler
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = metadata_df.copy()

    if embeddings.shape[1] >= 2:
        df['umap_x'] = embeddings[:, 0]
        df['umap_y'] = embeddings[:, 1]
    if embeddings.shape[1] >= 3:
        df['umap_z'] = embeddings[:, 2]

    df.to_csv(output_path, index=False)
    print(f"\nSaved embeddings to {output_path}")

    # Save parameters
    param_path = output_path.with_suffix('.json')
    with open(param_path, 'w') as f:
        json.dump(umap_params, f, indent=2)
    print(f"Saved parameters to {param_path}")

    # Save UMAP model
    if reducer is not None:
        model_path = output_path.with_suffix('.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'reducer': reducer,
                'scaler': scaler
            }, f)
        print(f"Saved UMAP model to {model_path}")

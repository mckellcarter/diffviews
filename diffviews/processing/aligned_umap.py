"""
AlignedUMAP computation for sigma-varying activation visualization.

Uses umap-learn's AlignedUMAP to create aligned 2D embeddings across
multiple sigma levels, enabling 3D visualization where Z = sigma.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .umap_backend import get_aligned_umap_class, to_numpy


def compute_aligned_umap(
    activations: np.ndarray,
    sigma_labels: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    alignment_regularisation: float = 0.01,
    alignment_window_size: int = 3,
    n_components: int = 2,
    normalize: bool = True,
    pca_components: Optional[int] = 50,
    random_state: Optional[int] = 42,
) -> Tuple[Dict[float, np.ndarray], Any, StandardScaler, Optional[PCA], Dict[float, NearestNeighbors], List[float]]:
    """
    Compute AlignedUMAP across sigma levels.

    Args:
        activations: (N_total, D) activation matrix with all sigma levels
        sigma_labels: (N_total,) sigma value per row
        n_neighbors: UMAP n_neighbors
        min_dist: UMAP min_dist
        alignment_regularisation: Higher = more alignment, less individual quality
        alignment_window_size: Forward/backward scope during alignment
        n_components: Output dimensions (usually 2)
        normalize: Whether to normalize before UMAP
        pca_components: PCA pre-reduction dims (None to skip)
        random_state: Random seed

    Returns:
        embeddings_per_sigma: {sigma: (N_per_sigma, 2)}
        aligned_mapper: Fitted AlignedUMAP
        scaler: Fitted StandardScaler
        pca_reducer: Fitted PCA (or None)
        nn_models: {sigma: NearestNeighbors} for trajectory projection
        sigma_levels: Sorted sigma levels (descending)
    """
    # Sort sigmas descending (high noise to low noise)
    unique_sigmas = np.unique(sigma_labels)
    sigma_levels = sorted(unique_sigmas, reverse=True)

    print(f"\nComputing AlignedUMAP across {len(sigma_levels)} sigma levels")
    print(f"  Sigmas: {sigma_levels}")
    print(f"  Total samples: {len(activations)}")

    # Normalize
    scaler = None
    if normalize:
        print("Normalizing activations...")
        scaler = StandardScaler()
        activations = scaler.fit_transform(activations)

    # PCA pre-reduction
    pca_reducer = None
    if pca_components and activations.shape[1] > pca_components:
        print(f"PCA reduction: {activations.shape[1]} -> {pca_components} dims")
        pca_reducer = PCA(n_components=pca_components, random_state=random_state)
        activations = pca_reducer.fit_transform(activations)
        print(f"  Explained variance: {pca_reducer.explained_variance_ratio_.sum():.2%}")

    # Split by sigma, preserving order within each slice
    datasets = []
    sigma_indices = {}  # Track original indices per sigma
    for sigma in sigma_levels:
        mask = sigma_labels == sigma
        indices = np.where(mask)[0]
        sigma_indices[sigma] = indices
        datasets.append(activations[indices])
        print(f"  Sigma {sigma}: {len(indices)} samples")

    # Verify same sample count per sigma (required for identity relations)
    counts = [len(d) for d in datasets]
    if len(set(counts)) != 1:
        raise ValueError(f"Unequal samples per sigma: {dict(zip(sigma_levels, counts))}")

    n_samples = counts[0]

    # Build identity relations (same sample order across all sigmas)
    relations = [{i: i for i in range(n_samples)} for _ in range(len(sigma_levels) - 1)]

    # Fit AlignedUMAP
    print(f"\nFitting AlignedUMAP (alignment_reg={alignment_regularisation})...")
    AlignedUMAP = get_aligned_umap_class()
    aligned_mapper = AlignedUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        random_state=random_state,
        verbose=True,
    )
    aligned_mapper.fit(datasets, relations=relations)

    # Extract embeddings and fit KNN models for projection
    embeddings_per_sigma = {}
    nn_models = {}
    for i, sigma in enumerate(sigma_levels):
        emb = to_numpy(aligned_mapper.embeddings_[i])
        embeddings_per_sigma[sigma] = emb
        print(f"  Sigma {sigma}: embeddings shape {emb.shape}")

        # Fit KNN on PCA-reduced activations for trajectory projection
        nn = NearestNeighbors(n_neighbors=min(15, n_samples), metric='euclidean')
        nn.fit(datasets[i])
        nn_models[sigma] = nn

    return embeddings_per_sigma, aligned_mapper, scaler, pca_reducer, nn_models, sigma_levels


def project_aligned_trajectory_point(
    activation: np.ndarray,
    sigma: float,
    scaler: Optional[StandardScaler],
    pca_reducer: Optional[PCA],
    nn_models: Dict[float, NearestNeighbors],
    embeddings_per_sigma: Dict[float, np.ndarray],
    sigma_levels: List[float],
    k: int = 5,
) -> Tuple[float, float]:
    """
    Project a trajectory point using k-NN interpolation.

    Since AlignedUMAP lacks transform(), we find nearest neighbors
    in activation space and interpolate their UMAP coordinates.

    Args:
        activation: (1, D) raw activation vector
        sigma: Sigma level of this trajectory point
        scaler: StandardScaler (or None)
        pca_reducer: PCA reducer (or None)
        nn_models: {sigma: NearestNeighbors}
        embeddings_per_sigma: {sigma: (N, 2)}
        sigma_levels: Sorted sigma levels
        k: Number of neighbors for interpolation

    Returns:
        (x, y) UMAP coordinates
    """
    # Find closest sigma level
    closest_sigma = min(sigma_levels, key=lambda s: abs(np.log(s + 1e-8) - np.log(sigma + 1e-8)))

    # Preprocess activation
    act = activation.reshape(1, -1)
    if scaler is not None:
        act = scaler.transform(act)
    if pca_reducer is not None:
        act = pca_reducer.transform(act)

    # Find k nearest neighbors
    nn = nn_models[closest_sigma]
    k_actual = min(k, nn.n_samples_fit_)
    distances, indices = nn.kneighbors(act, n_neighbors=k_actual)

    # Distance-weighted interpolation
    distances = distances[0]
    indices = indices[0]

    # Avoid division by zero
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum()

    neighbor_coords = embeddings_per_sigma[closest_sigma][indices]
    projected = np.average(neighbor_coords, axis=0, weights=weights)

    return float(projected[0]), float(projected[1])


def save_aligned_embeddings(
    embeddings_per_sigma: Dict[float, np.ndarray],
    metadata_df: pd.DataFrame,
    output_dir: Path,
    scaler: Optional[StandardScaler],
    pca_reducer: Optional[PCA],
    nn_models: Dict[float, NearestNeighbors],
    sigma_levels: List[float],
    umap_params: dict,
):
    """
    Save AlignedUMAP results to disk.

    Creates:
        - embeddings.csv: Long-form CSV with sigma column
        - embeddings.json: Parameters
        - embeddings.pkl: Pickled models (scaler, pca, nn_models, embeddings)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build long-form CSV: one row per sample per sigma
    # Assumes metadata_df has one row per sample (not per sigma)
    n_samples = len(embeddings_per_sigma[sigma_levels[0]])

    rows = []
    for sigma in sigma_levels:
        emb = embeddings_per_sigma[sigma]
        for i in range(n_samples):
            row = {'sigma': sigma, 'umap_x': emb[i, 0], 'umap_y': emb[i, 1]}
            # Add metadata from the corresponding row
            # Metadata is assumed to be ordered consistently
            if i < len(metadata_df):
                meta_row = metadata_df.iloc[i]
                for col in ['class_label', 'class_name', 'image_path']:
                    if col in meta_row:
                        row[col] = meta_row[col]
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / 'embeddings.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved embeddings to {csv_path}")

    # Save params
    params = {
        **umap_params,
        'sigma_levels': sigma_levels,
        'n_samples_per_sigma': n_samples,
        'type': 'aligned_3d',
    }
    json_path = output_dir / 'embeddings.json'
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved parameters to {json_path}")

    # Save models (skip aligned_mapper - contains unpicklable numba objects)
    pkl_path = output_dir / 'embeddings.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'pca_reducer': pca_reducer,
            'nn_models': nn_models,
            'sigma_levels': sigma_levels,
            'embeddings_per_sigma': embeddings_per_sigma,
        }, f)
    print(f"Saved models to {pkl_path}")


def load_aligned_embeddings(
    embeddings_dir: Path,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Load AlignedUMAP data from disk.

    Returns:
        df: Long-form DataFrame with sigma, umap_x, umap_y columns
        params: UMAP parameters dict
        model_data: Dict with aligned_mapper, scaler, pca_reducer, nn_models, etc.
    """
    embeddings_dir = Path(embeddings_dir)

    csv_path = embeddings_dir / 'embeddings.csv'
    df = pd.read_csv(csv_path)

    json_path = embeddings_dir / 'embeddings.json'
    with open(json_path) as f:
        params = json.load(f)

    pkl_path = embeddings_dir / 'embeddings.pkl'
    with open(pkl_path, 'rb') as f:
        model_data = pickle.load(f)

    return df, params, model_data

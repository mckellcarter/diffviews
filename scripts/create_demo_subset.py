#!/usr/bin/env python
"""Create demo subset from full dataset."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from diffviews.processing.umap import compute_umap, save_embeddings
from diffviews.core.extractor import load_activations, flatten_activations


# Diverse classes for demo
DEFAULT_CLASSES = [
    2,    # great_white_shark
    67,   # diamondback
    86,   # partridge
    88,   # macaw
    115,  # sea_slug
    173,  # Ibizan_hound
    197,  # giant_schnauzer
    319,  # dragonfly
    394,  # sturgeon
    649,  # megalith
    665,  # moped
    704,  # parking_meter
    725,  # pitcher
    751,  # racer
    827,  # stove
    930,  # French_loaf
]


def create_demo_subset(
    source_dir: Path,
    source_csv: Path,
    output_dir: Path,
    classes: list,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
):
    """Create demo subset with images, activations, and UMAP."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {classes}")

    # Read full CSV
    print("\nLoading embeddings CSV...")
    df = pd.read_csv(source_csv)

    # Filter to selected classes
    subset_df = df[df['class_label'].isin(classes)].copy()
    print(f"Filtered to {len(subset_df)} samples from {len(classes)} classes")

    # Create output directories
    (output_dir / "images" / "imagenet_real").mkdir(parents=True, exist_ok=True)
    (output_dir / "activations" / "imagenet_real").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "imagenet_real").mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    # Copy images
    print("\nCopying images...")
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        src = source_dir / row['image_path']
        dst = output_dir / row['image_path']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Load and save activations (need to reindex)
    print("\nLoading activations for UMAP training...")

    # Group by batch file to load efficiently
    batch_groups = subset_df.groupby('activation_path')

    all_activations = []
    new_metadata = []
    sample_idx = 0

    for batch_path, group in tqdm(batch_groups, desc="Loading batches"):
        full_batch_path = source_dir / batch_path
        if not full_batch_path.with_suffix('.npz').exists():
            print(f"Warning: Missing {full_batch_path}")
            continue

        activations, _ = load_activations(full_batch_path)

        for _, row in group.iterrows():
            batch_idx = row['batch_index']
            flat = flatten_activations(activations)
            all_activations.append(flat[batch_idx])

            new_metadata.append({
                'sample_id': row['sample_id'],
                'class_label': int(row['class_label']),
                'class_name': row['class_name'],
                'image_path': row['image_path'],
                'conditioning_sigma': row.get('conditioning_sigma'),
            })
            sample_idx += 1

    activation_matrix = np.stack(all_activations, axis=0)
    print(f"Activation matrix: {activation_matrix.shape}")

    # Save activations as single batch
    print("\nSaving activations...")
    act_output = output_dir / "activations" / "imagenet_real" / "batch_000000.npz"

    # Determine layer shapes from original
    first_batch = source_dir / subset_df.iloc[0]['activation_path']
    orig_acts, _ = load_activations(first_batch)

    # Split back into layer format for saving
    layer_shapes = {}
    offset = 0
    act_dict = {}
    for layer_name in sorted(orig_acts.keys()):
        layer_shape = orig_acts[layer_name].shape[1:]  # (C, H, W)
        layer_shapes[layer_name] = layer_shape
        layer_size = np.prod(layer_shape)
        layer_acts = activation_matrix[:, offset:offset+layer_size]
        layer_acts = layer_acts.reshape(-1, *layer_shape)
        act_dict[layer_name] = layer_acts.astype(np.float32)
        offset += layer_size

    np.savez_compressed(act_output, **act_dict)
    print(f"Saved activations to {act_output}")

    # Update metadata with new batch index
    for i, m in enumerate(new_metadata):
        m['activation_path'] = "activations/imagenet_real/batch_000000"
        m['batch_index'] = i

    # Save metadata
    metadata_output = output_dir / "metadata" / "imagenet_real" / "dataset_info.json"
    with open(metadata_output, 'w') as f:
        json.dump({
            'samples': new_metadata,
            'layer_shapes': {k: list(v) for k, v in layer_shapes.items()},
            'num_samples': len(new_metadata),
        }, f, indent=2)
    print(f"Saved metadata to {metadata_output}")

    # Train new UMAP
    print("\nTraining UMAP on subset...")
    embeddings, reducer, scaler = compute_umap(
        activation_matrix,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
    )

    # Create output dataframe
    meta_df = pd.DataFrame(new_metadata)

    # Save embeddings
    umap_params = {
        'n_neighbors': umap_n_neighbors,
        'min_dist': umap_min_dist,
        'layers': list(layer_shapes.keys()),
        'model': 'imagenet_real',
    }

    output_csv = output_dir / "embeddings" / "demo_embeddings.csv"
    save_embeddings(
        embeddings,
        meta_df,
        output_csv,
        umap_params,
        reducer=reducer,
        scaler=scaler,
    )

    print(f"\nDemo subset created at {output_dir}")
    print(f"Run with:")
    print(f"  python -m diffviews.visualization.app --data_dir {output_dir} --embeddings {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Create demo subset")
    parser.add_argument("--source_dir", type=str, required=True, help="Source data directory")
    parser.add_argument("--source_csv", type=str, required=True, help="Source embeddings CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES, help="Class IDs to include")
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    args = parser.parse_args()

    create_demo_subset(
        source_dir=Path(args.source_dir),
        source_csv=Path(args.source_csv),
        output_dir=Path(args.output_dir),
        classes=args.classes,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )


if __name__ == "__main__":
    main()

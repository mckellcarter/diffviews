#!/usr/bin/env python
"""Extract activations from diffusion models for visualization.

Supports all adapters: DMD2, EDM (ImageNet), and MSCOCO (text-to-image).
Uses adapter interface for automatic configuration.

Usage:
    # DMD2 (ImageNet, class-conditioned)
    python scripts/extract_activations.py \
        --adapter dmd2-imagenet-64 \
        --image_dir /Volumes/diffattr_external/datasets/imagenet/train \
        --checkpoint data/dmd2/checkpoints/dmd2-imagenet-64-10step.pkl \
        --output_dir data/dmd2 \
        --num_samples 300

    # EDM (ImageNet, class-conditioned)
    python scripts/extract_activations.py \
        --adapter edm-imagenet-64 \
        --image_dir /Volumes/diffattr_external/datasets/imagenet/train \
        --checkpoint data/edm/checkpoints/edm-imagenet-64x64-cond-adm.pkl \
        --output_dir data/edm \
        --num_samples 300

    # MSCOCO (text-conditioned)
    python scripts/extract_activations.py \
        --adapter mscoco-t2i-128 \
        --image_dir /Volumes/diffattr_external/datasets/coco/train2017 \
        --captions /Volumes/diffattr_external/datasets/coco/annotations/captions_train2017.json \
        --checkpoint /path/to/mscoco/model.bin \
        --output_dir data/mscoco \
        --num_samples 300
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from adapt_diff import get_adapter, ActivationExtractor


# 5 noise levels as percentages (0=clean, 100=pure noise)
# Matches existing 4-point range + adds clean point at 0%
DEFAULT_NOISE_LEVELS = [100.0, 82.0, 54.0, 35.0, 0.0]

# Default layer to extract
DEFAULT_LAYER = "midblock"


def find_existing_images(
    output_dir: Path,
    dataset_type: str
) -> List[Tuple[Path, int, str]]:
    """Find pre-extracted images in output directory.

    Returns list of (image_path, class_idx, class_name) for compatibility.
    Class info loaded from existing metadata if available.
    """
    images_dir = output_dir / "images" / dataset_type
    metadata_path = output_dir / "metadata" / dataset_type / "dataset_info.json"

    # Load existing metadata for class labels
    class_info = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            for sample in meta.get("samples", []):
                # Extract sample index from sample_id
                img_name = Path(sample.get("image_path", "")).name
                if "class_label" in sample:
                    class_info[img_name] = (sample["class_label"], sample.get("class_name", "unknown"))
                elif "caption" in sample:
                    class_info[img_name] = (None, sample["caption"])

    # Find all images
    images = []
    for img_path in sorted(images_dir.glob("*.png")):
        if img_path.name in class_info:
            label, name = class_info[img_path.name]
            images.append((img_path, label, name))
        else:
            # Default if no metadata
            images.append((img_path, 0, "unknown"))

    # Deduplicate - we only want unique images, not per-noise-level entries
    seen = set()
    unique_images = []
    for img_path, label, name in images:
        if img_path.name not in seen:
            seen.add(img_path.name)
            unique_images.append((img_path, label, name))

    print(f"Found {len(unique_images)} existing images in {images_dir}")
    return unique_images


def load_captions(captions_path: Path) -> Tuple[Dict, Dict]:
    """Load COCO captions."""
    with open(captions_path) as f:
        data = json.load(f)

    captions_by_image = defaultdict(list)
    for ann in data['annotations']:
        captions_by_image[ann['image_id']].append(ann['caption'])

    filename_to_id = {}
    for img in data['images']:
        filename_to_id[img['file_name']] = img['id']

    return captions_by_image, filename_to_id


def find_imagenet_images(
    image_dir: Path,
    num_samples: int,
    seed: int = 42
) -> List[Tuple[Path, int, str]]:
    """Find ImageNet images with class labels.

    Returns list of (image_path, class_idx, class_name).
    """
    random.seed(seed)

    # ImageNet structure: image_dir/n01234567/n01234567_123.JPEG
    synset_dirs = [d for d in image_dir.iterdir() if d.is_dir() and d.name.startswith('n')]

    if not synset_dirs:
        raise ValueError(f"No synset directories found in {image_dir}")

    # Load synset to class index mapping
    synset_to_idx = {}
    try:
        labels_path = Path(__file__).parent.parent / "data" / "imagenet_standard_class_index.json"
        if labels_path.exists():
            with open(labels_path) as f:
                idx_to_synset = json.load(f)
                for idx, (synset, name) in idx_to_synset.items():
                    synset_to_idx[synset] = (int(idx), name)
    except Exception:
        pass

    # Collect all images
    all_images = []
    for synset_dir in synset_dirs:
        synset = synset_dir.name
        class_idx, class_name = synset_to_idx.get(synset, (0, synset))

        for ext in ["*.JPEG", "*.jpeg", "*.png", "*.jpg"]:
            for img_path in synset_dir.glob(ext):
                all_images.append((img_path, class_idx, class_name))

    print(f"Found {len(all_images)} images in {len(synset_dirs)} classes")

    if len(all_images) < num_samples:
        print(f"Warning: only {len(all_images)} images available")
        num_samples = len(all_images)

    return random.sample(all_images, num_samples)


def find_coco_images(
    image_dir: Path,
    captions_path: Path,
    num_samples: int,
    seed: int = 42
) -> List[Tuple[Path, str]]:
    """Find COCO images with captions.

    Returns list of (image_path, caption).
    """
    random.seed(seed)

    captions_by_image, filename_to_id = load_captions(captions_path)

    # Find images with captions
    image_paths = []
    for p in image_dir.glob("*.jpg"):
        if p.name in filename_to_id:
            image_paths.append(p)

    print(f"Found {len(image_paths)} images with captions")

    if len(image_paths) < num_samples:
        print(f"Warning: only {len(image_paths)} images available")
        num_samples = len(image_paths)

    selected = random.sample(image_paths, num_samples)

    # Pair with captions
    result = []
    for img_path in selected:
        image_id = filename_to_id[img_path.name]
        caption = random.choice(captions_by_image[image_id])
        result.append((img_path, caption))

    return result


def load_and_preprocess_image(image_path: Path, resolution: int) -> torch.Tensor:
    """Load and preprocess image to [-1, 1] tensor."""
    img = Image.open(image_path).convert("RGB")

    # Center crop to square
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Resize
    img = img.resize((resolution, resolution), Image.LANCZOS)

    # Convert to tensor [-1, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    img_tensor = img_tensor * 2.0 - 1.0

    return img_tensor


def add_noise_at_level(
    x: torch.Tensor,
    adapter,
    noise_level: float,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Add noise to input at specified noise level (0-100).

    Returns (noised_input, native_timestep, sigma_for_metadata).
    """
    native = adapter.noise_level_to_native(torch.tensor(noise_level))

    if hasattr(adapter, 'SIGMA_MIN'):
        # Sigma-based (DMD2/EDM): x_noised = x + noise * sigma
        sigma = float(native)
        noise = torch.randn_like(x)
        noised = x + noise * sigma
        native_tensor = torch.tensor([sigma], device=device, dtype=x.dtype)
        return noised, native_tensor, sigma
    else:
        # Timestep-based (MSCOCO): use scheduler alphas
        timestep = int(native.item())
        noise = torch.randn_like(x)

        alpha_cumprod = adapter.scheduler.alphas_cumprod[timestep]
        sqrt_alpha = alpha_cumprod.sqrt()
        sqrt_one_minus_alpha = (1 - alpha_cumprod).sqrt()

        noised = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        native_tensor = torch.tensor([timestep], device=device, dtype=torch.long)

        # Compute equivalent sigma for metadata
        sigma = ((1 - alpha_cumprod) / alpha_cumprod).sqrt().item()
        return noised, native_tensor, sigma


def extract_activations(
    adapter_name: str,
    image_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    captions_path: Optional[Path] = None,
    num_samples: int = 300,
    noise_levels: List[float] = None,
    layer: str = DEFAULT_LAYER,
    device: str = "cuda",
    seed: int = 42,
    use_existing_images: bool = False,
):
    """Extract activations from images at multiple noise levels."""

    noise_levels = noise_levels or DEFAULT_NOISE_LEVELS
    random.seed(seed)

    # Load adapter
    print(f"Loading adapter '{adapter_name}' from {checkpoint_path}...")
    AdapterClass = get_adapter(adapter_name)
    adapter = AdapterClass.from_checkpoint(str(checkpoint_path), device=device)

    resolution = adapter.resolution
    conditioning_type = adapter.conditioning_type
    uses_latent = adapter.uses_latent

    print(f"Resolution: {resolution}x{resolution}")
    print(f"Conditioning: {conditioning_type}")
    print(f"Latent space: {uses_latent}")
    print(f"Noise levels (%): {noise_levels}")

    # Validate layer
    layer_shapes = adapter.get_layer_shapes()
    if layer not in layer_shapes:
        print(f"Available layers: {list(layer_shapes.keys())}")
        raise ValueError(f"Layer '{layer}' not found")
    layer_shape = layer_shapes[layer]
    print(f"Layer '{layer}' shape: {layer_shape}")

    # Find images based on conditioning type
    dataset_type = "imagenet_real" if conditioning_type == 'class' else "train2017"

    if use_existing_images:
        # Use pre-extracted images from output directory
        images_data = find_existing_images(Path(output_dir), dataset_type)
        if num_samples < len(images_data):
            images_data = images_data[:num_samples]
    elif conditioning_type == 'class':
        images_data = find_imagenet_images(image_dir, num_samples, seed)
    else:
        if captions_path is None:
            raise ValueError("--captions required for text-conditioned models")
        images_data = find_coco_images(image_dir, captions_path, num_samples, seed)

    # Create output dirs
    output_dir = Path(output_dir)
    (output_dir / "activations" / dataset_type).mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / dataset_type).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / dataset_type).mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Extract
    all_activations = []
    metadata_samples = []
    sample_idx = 0

    extractor = ActivationExtractor(adapter, layers=[layer])
    extractor.register_hooks()

    try:
        for item in tqdm(images_data, desc="Extracting"):
            if conditioning_type == 'class':
                img_path, class_idx, class_name = item
                caption = None
            else:
                img_path, caption = item
                class_idx, class_name = None, None

            # Load image
            try:
                img_tensor = load_and_preprocess_image(img_path, resolution)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Encode to latent if needed
            if uses_latent:
                with torch.no_grad():
                    img_tensor = adapter.encode(img_tensor)

            # Save preprocessed image (skip if using existing images)
            if not use_existing_images:
                img_save_path = output_dir / "images" / dataset_type / f"sample_{sample_idx:06d}.png"
                img_pil = Image.open(img_path).convert("RGB")
                w, h = img_pil.size
                size = min(w, h)
                left = (w - size) // 2
                top = (h - size) // 2
                img_pil = img_pil.crop((left, top, left + size, top + size))
                img_pil = img_pil.resize((resolution, resolution), Image.LANCZOS)
                img_pil.save(img_save_path)

            # Prepare conditioning using adapter
            if conditioning_type == 'class':
                cond = adapter.prepare_conditioning(
                    class_label=class_idx,
                    batch_size=1,
                    device=device
                )
            else:
                cond = adapter.prepare_conditioning(
                    text=caption,
                    batch_size=1,
                    device=device
                )

            # Extract at each noise level
            for noise_level in noise_levels:
                extractor.clear()

                # Add noise at this level
                noised, native_t, sigma = add_noise_at_level(
                    img_tensor, adapter, noise_level, device
                )

                # Forward pass
                with torch.no_grad():
                    if conditioning_type == 'class':
                        _ = adapter.forward(noised, native_t, class_labels=cond)
                    else:
                        _ = adapter.forward(noised, native_t, **cond)

                # Get activation
                acts = extractor.get_activations()
                act = acts[layer].cpu().numpy()

                if len(act.shape) == 4:
                    act = act.reshape(act.shape[0], -1)

                all_activations.append(act[0])

                # Build metadata
                if use_existing_images:
                    image_rel_path = f"images/{dataset_type}/{img_path.name}"
                else:
                    image_rel_path = f"images/{dataset_type}/sample_{sample_idx:06d}.png"

                sample_meta = {
                    "sample_id": f"sample_{sample_idx:06d}_n{int(noise_level)}",
                    "image_path": image_rel_path,
                    "original_path": str(img_path.name),
                    "noise_level": noise_level,
                    "conditioning_sigma": sigma,
                    "activation_path": f"activations/{dataset_type}/batch_000000",
                    "batch_index": len(all_activations) - 1,
                }

                if conditioning_type == 'class':
                    sample_meta["class_label"] = class_idx
                    sample_meta["class_name"] = class_name
                else:
                    sample_meta["caption"] = caption

                metadata_samples.append(sample_meta)

            sample_idx += 1

    finally:
        extractor.remove_hooks()

    # Save activations
    activation_matrix = np.stack(all_activations, axis=0).astype(np.float32)
    print(f"Activation matrix: {activation_matrix.shape}")

    act_path = output_dir / "activations" / dataset_type / "batch_000000.npz"
    np.savez_compressed(act_path, **{layer: activation_matrix})
    print(f"Saved: {act_path}")

    npy_path = act_path.with_suffix(".npy")
    np.save(npy_path, activation_matrix)
    print(f"Saved: {npy_path}")

    # Save npy metadata
    npy_json_path = Path(str(npy_path) + ".json")
    npy_info = {
        "layers": [layer],
        "shapes": {layer: list(layer_shape)},
        "total_features": activation_matrix.shape[1],
        "num_samples": activation_matrix.shape[0],
    }
    with open(npy_json_path, "w") as f:
        json.dump(npy_info, f, indent=2)

    # Save dataset metadata
    metadata = {
        "samples": metadata_samples,
        "layer_shapes": {layer: list(layer_shape)},
        "num_samples": len(metadata_samples),
        "num_images": sample_idx,
        "noise_levels": noise_levels,
        "layer": layer,
    }

    metadata_path = output_dir / "metadata" / dataset_type / "dataset_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # Compute sigma range from actual values
    sigmas = [s["conditioning_sigma"] for s in metadata_samples]
    sigma_min = min(sigmas)
    sigma_max = max(sigmas)

    # Save config.json
    config = {
        "adapter": adapter_name,
        "checkpoint": f"checkpoints/{checkpoint_path.name}",
        "default_steps": 20,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "dataset_type": dataset_type,
        "conditioning_type": conditioning_type,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    print(f"\nExtraction complete!")
    print(f"  Images: {sample_idx}")
    print(f"  Noise levels: {len(noise_levels)}")
    print(f"  Total samples: {len(metadata_samples)}")
    print(f"  Sigma range: {sigma_min:.4f} - {sigma_max:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from diffusion models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--adapter", type=str, required=True,
                        choices=["dmd2-imagenet-64", "edm-imagenet-64", "mscoco-t2i-128"],
                        help="Adapter name")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Image directory (ImageNet train/ or COCO train2017/). Not required with --use_existing_images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--captions", type=str, default=None,
                        help="COCO captions JSON (required for MSCOCO)")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of images to sample")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS,
                        help="Noise levels as percentages (0=clean, 100=noise)")
    parser.add_argument("--layer", type=str, default=DEFAULT_LAYER,
                        help="Layer to extract")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (cuda, mps, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_existing_images", action="store_true",
                        help="Use pre-extracted images from output_dir instead of source")

    args = parser.parse_args()

    extract_activations(
        adapter_name=args.adapter,
        image_dir=Path(args.image_dir) if args.image_dir else None,
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        captions_path=Path(args.captions) if args.captions else None,
        num_samples=args.num_samples,
        noise_levels=args.noise_levels,
        layer=args.layer,
        device=args.device,
        seed=args.seed,
        use_existing_images=args.use_existing_images,
    )


if __name__ == "__main__":
    main()

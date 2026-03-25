#!/usr/bin/env python
"""Extract activations from MSCOCO T2I model for visualization.

Uses real COCO captions encoded with CLIP ViT-L/14 to match training.

Usage:
    python scripts/extract_mscoco_activations.py \
        --image_dir /Volumes/diffattr_external/datasets/coco/train2017 \
        --captions /Volumes/diffattr_external/datasets/coco/annotations/captions_train2017.json \
        --checkpoint /Users/mckell/Documents/GitHub/adapt_diff/checkpoints/mscoco/model.bin \
        --output_dir data/mscoco \
        --num_samples 300 \
        --device mps
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from adapt_diff import get_adapter
from diffviews.core.extractor import ActivationExtractor


# Timesteps matching DMD2 sigma schedule (80, 14, 1.9, 0.62)
# High to low noise
DEFAULT_TIMESTEPS = [930, 721, 386, 175]

# Layer to extract
EXTRACT_LAYER = "mid_block"


def load_clip_model(device: str):
    """Load OpenCLIP ViT-H/14 for text encoding (1024-dim, matches SD2.x)."""
    try:
        import open_clip
        # ViT-H/14 produces 1024-dim embeddings matching the model's cross_attention_dim
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k'
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        print("Loaded OpenCLIP ViT-H/14 (1024-dim)")
        return model, tokenizer, "open_clip"
    except Exception as e:
        print(f"open_clip ViT-H failed: {e}")
        print("Falling back to ViT-L with projection...")
        from transformers import CLIPTextModel, CLIPTokenizer
        model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        return model, tokenizer, "transformers"


def encode_text_open_clip(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Encode text using open_clip, returning full sequence embeddings."""
    import open_clip
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        # Use encode_text with normalize=False to get raw features
        # This returns CLS token embedding (1, 1024)
        text_features = model.encode_text(tokens, normalize=False)
        # Expand to sequence format (1, 1, 1024) for cross-attention
        text_emb = text_features.unsqueeze(1)
    return text_emb.float()


def encode_text_transformers(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Encode text using transformers CLIP."""
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use last_hidden_state for full sequence (1, seq_len, 768)
        hidden_states = outputs.last_hidden_state
    return hidden_states


def load_captions(captions_path: Path) -> dict:
    """Load COCO captions, return dict mapping image_id -> list of captions."""
    with open(captions_path) as f:
        data = json.load(f)

    captions_by_image = defaultdict(list)
    for ann in data['annotations']:
        captions_by_image[ann['image_id']].append(ann['caption'])

    # Also build filename -> image_id mapping
    filename_to_id = {}
    for img in data['images']:
        filename_to_id[img['file_name']] = img['id']

    return captions_by_image, filename_to_id


def load_and_preprocess_image(image_path: Path, resolution: int = 128) -> torch.Tensor:
    """Load image and preprocess for the model."""
    img = Image.open(image_path).convert("RGB")

    # Center crop to square
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Resize to model resolution
    img = img.resize((resolution, resolution), Image.LANCZOS)

    # Convert to tensor [-1, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

    return img_tensor


def extract_activations(
    image_dir: Path,
    captions_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    num_samples: int = 300,
    timesteps: list = None,
    layer: str = EXTRACT_LAYER,
    device: str = "cuda",
    seed: int = 42,
    use_null_text: bool = False,
):
    """Extract activations from images at multiple timesteps."""

    timesteps = timesteps or DEFAULT_TIMESTEPS
    random.seed(seed)

    # Load captions
    print(f"Loading captions from {captions_path}...")
    captions_by_image, filename_to_id = load_captions(captions_path)
    print(f"Loaded captions for {len(captions_by_image)} images")

    # Load CLIP if using real captions
    clip_model = None
    clip_tokenizer = None
    clip_type = None
    if not use_null_text:
        print("Loading CLIP for text encoding...")
        clip_model, clip_tokenizer, clip_type = load_clip_model(device)

    print(f"Loading adapter from {checkpoint_path}...")
    AdapterClass = get_adapter("mscoco-t2i-128")
    adapter = AdapterClass.from_checkpoint(str(checkpoint_path), device=device)

    resolution = adapter.resolution
    print(f"Model resolution: {resolution}x{resolution}")
    print(f"Extracting layer: {layer}")
    print(f"Timesteps: {timesteps}")

    # Get layer shape
    layer_shapes = adapter.get_layer_shapes()
    if layer not in layer_shapes:
        print(f"Available layers: {list(layer_shapes.keys())}")
        raise ValueError(f"Layer {layer} not found")
    layer_shape = layer_shapes[layer]
    print(f"Layer shape: {layer_shape}")

    # Find all images that have captions
    image_paths = []
    for p in image_dir.glob("*.jpg"):
        if p.name in filename_to_id:
            image_paths.append(p)

    print(f"Found {len(image_paths)} images with captions")

    if len(image_paths) < num_samples:
        print(f"Warning: Only {len(image_paths)} images available")
        num_samples = len(image_paths)

    # Random sample
    selected_paths = random.sample(image_paths, num_samples)
    print(f"Selected {num_samples} images")

    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / "activations" / "train2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "train2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "train2017").mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Collect all activations
    all_activations = []
    metadata_samples = []
    sample_idx = 0

    # Create extractor
    extractor = ActivationExtractor(adapter, layers=[layer])
    extractor.register_hooks()

    try:
        for img_path in tqdm(selected_paths, desc="Extracting"):
            # Load and preprocess
            try:
                img_tensor = load_and_preprocess_image(img_path, resolution)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dim

            # Get caption for this image
            image_id = filename_to_id[img_path.name]
            captions = captions_by_image[image_id]
            caption = random.choice(captions)  # Random caption

            # Encode text
            if use_null_text:
                text_emb = torch.zeros(1, 1, 1024, device=device)
            else:
                if clip_type == "open_clip":
                    text_emb = encode_text_open_clip(
                        clip_model, clip_tokenizer, caption, device
                    )
                else:
                    text_emb = encode_text_transformers(
                        clip_model, clip_tokenizer, caption, device
                    )
                    # Pad to 1024 if using ViT-L (768 dim)
                    if text_emb.shape[-1] != 1024:
                        pad_size = 1024 - text_emb.shape[-1]
                        text_emb = torch.nn.functional.pad(text_emb, (0, pad_size))

            # Encode to latent space
            with torch.no_grad():
                latent = adapter.encode_images(img_tensor)

            # Save preprocessed image
            img_save_path = output_dir / "images" / "train2017" / f"sample_{sample_idx:06d}.png"
            img_pil = Image.open(img_path).convert("RGB")
            w, h = img_pil.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            img_pil = img_pil.crop((left, top, left + size, top + size))
            img_pil = img_pil.resize((resolution, resolution), Image.LANCZOS)
            img_pil.save(img_save_path)

            # Extract at each timestep
            for t in timesteps:
                extractor.clear()

                # Add noise at timestep t
                timestep_tensor = torch.tensor([t], device=device, dtype=torch.long)
                noise = torch.randn_like(latent)

                # Get alpha for this timestep
                alpha_cumprod = adapter.scheduler.alphas_cumprod[t]
                sqrt_alpha = alpha_cumprod.sqrt()
                sqrt_one_minus_alpha = (1 - alpha_cumprod).sqrt()

                noisy_latent = sqrt_alpha * latent + sqrt_one_minus_alpha * noise

                # Forward pass with text embeddings
                with torch.no_grad():
                    _ = adapter.forward(
                        noisy_latent,
                        timestep_tensor,
                        encoder_hidden_states=text_emb,
                    )

                # Get activation
                acts = extractor.get_activations()
                act = acts[layer].cpu().numpy()

                # Flatten spatial dims
                if len(act.shape) == 4:
                    act = act.reshape(act.shape[0], -1)

                all_activations.append(act[0])  # Remove batch dim

                # Compute sigma for metadata
                sigma = ((1 - alpha_cumprod) / alpha_cumprod).sqrt().item()

                metadata_samples.append({
                    "sample_id": f"sample_{sample_idx:06d}_t{t}",
                    "image_path": f"images/train2017/sample_{sample_idx:06d}.png",
                    "original_path": str(img_path.name),
                    "caption": caption,
                    "timestep": t,
                    "conditioning_sigma": sigma,
                    "activation_path": "activations/train2017/batch_000000",
                    "batch_index": len(all_activations) - 1,
                })

            sample_idx += 1

    finally:
        extractor.remove_hooks()

    # Stack activations
    activation_matrix = np.stack(all_activations, axis=0).astype(np.float32)
    print(f"Activation matrix: {activation_matrix.shape}")

    # Save activations
    act_path = output_dir / "activations" / "train2017" / "batch_000000.npz"
    np.savez_compressed(act_path, **{layer: activation_matrix})
    print(f"Saved activations to {act_path}")

    # Also save as .npy for fast loading
    npy_path = act_path.with_suffix(".npy")
    np.save(npy_path, activation_matrix)

    # Save metadata
    metadata = {
        "samples": metadata_samples,
        "layer_shapes": {layer: list(layer_shape)},
        "num_samples": len(metadata_samples),
        "num_images": sample_idx,
        "timesteps": timesteps,
        "layer": layer,
    }

    metadata_path = output_dir / "metadata" / "train2017" / "dataset_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Create config.json (sigmas matching DMD2 schedule)
    config = {
        "adapter": "mscoco-t2i-128",
        "checkpoint": "checkpoints/mscoco-t2i-128.bin",
        "default_steps": 20,
        "sigma_max": 80.0,
        "sigma_min": 0.62,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    print(f"\nExtraction complete!")
    print(f"  Images: {sample_idx}")
    print(f"  Timesteps: {len(timesteps)}")
    print(f"  Total samples: {len(metadata_samples)}")
    print(f"\nNext: Run UMAP computation on the activations")


def main():
    parser = argparse.ArgumentParser(description="Extract MSCOCO activations")
    parser.add_argument("--image_dir", type=str, required=True, help="COCO image directory")
    parser.add_argument("--captions", type=str, required=True, help="COCO captions JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="data/mscoco", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of images to sample")
    parser.add_argument("--timesteps", type=int, nargs="+", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--layer", type=str, default=EXTRACT_LAYER)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null_text", action="store_true", help="Use null text embeddings")
    args = parser.parse_args()

    extract_activations(
        image_dir=Path(args.image_dir),
        captions_path=Path(args.captions),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        timesteps=args.timesteps,
        layer=args.layer,
        device=args.device,
        seed=args.seed,
        use_null_text=args.null_text,
    )


if __name__ == "__main__":
    main()

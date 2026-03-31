"""
Image generation with activation masking using adapter interface.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from adapt_diff import GeneratorAdapter
from .extractor import ActivationExtractor
from .masking import ActivationMasker


def tensor_to_uint8_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor in [-1, 1] to uint8 in [0, 255].

    Args:
        tensor: Image tensor (B, C, H, W) in range [-1, 1]

    Returns:
        uint8 tensor (B, H, W, C) in range [0, 255]
    """
    images = ((tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu()
    return images


@torch.no_grad()
def generate_with_mask(
    adapter: GeneratorAdapter,
    masker: ActivationMasker,
    class_label: Optional[int] = None,
    conditioning_sigma: float = 0.1,
    num_samples: int = 1,
    device: str = 'cuda',
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate images with fixed activations (single-step).

    Args:
        adapter: GeneratorAdapter instance
        masker: ActivationMasker with masks set and hooks registered
        class_label: Class label (0-999), random if None, -1 for uniform
        conditioning_sigma: Noise level
        num_samples: Number of images
        device: Device for generation
        seed: Random seed

    Returns:
        (images, labels) - images as (N, H, W, 3) uint8
    """
    if seed is not None:
        torch.manual_seed(seed)

    resolution = adapter.resolution
    num_classes = adapter.num_classes

    # Generate labels
    if class_label is not None and class_label < 0:
        random_labels = torch.tensor([-1], device=device).repeat(num_samples)
        one_hot = torch.ones((num_samples, num_classes), device=device) / num_classes
    elif class_label is None:
        random_labels = torch.randint(0, num_classes, (num_samples,), device=device)
        one_hot = torch.eye(num_classes, device=device)[random_labels]
    else:
        random_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
        one_hot = torch.eye(num_classes, device=device)[random_labels]

    # Generate noise
    noise = torch.randn(num_samples, 3, resolution, resolution, device=device)
    sigma_tensor = torch.ones(num_samples, device=device) * conditioning_sigma

    # Generate with masked activations
    generated = adapter.forward(noise * conditioning_sigma, sigma_tensor, one_hot)

    # Decode latents to pixel space (identity for pixel-space models)
    generated = adapter.decode(generated)

    # Convert to uint8
    images = tensor_to_uint8_image(generated)

    return images, random_labels.cpu()


def _prepare_conditioning(
    adapter: GeneratorAdapter,
    class_label: Optional[int],
    caption: Optional[str],
    num_samples: int,
    device: str
) -> Tuple[any, any, torch.Tensor]:
    """
    Prepare conditioning and unconditioning for CFG.

    Returns:
        (cond, uncond, random_labels) - cond/uncond format depends on model type
    """
    num_classes = adapter.num_classes
    random_labels = torch.zeros(num_samples, device=device, dtype=torch.long)

    if caption is not None:
        # T2I model - use adapter's prepare_conditioning for proper encoding
        cond = adapter.prepare_conditioning(text=caption, batch_size=num_samples)
        uncond = adapter.prepare_conditioning(text="", batch_size=num_samples)
        return cond, uncond, random_labels

    # Class-conditioned model
    if class_label is not None and class_label < 0:
        # Uniform distribution (unconditional)
        random_labels = torch.tensor([-1], device=device).repeat(num_samples)
        one_hot = torch.ones((num_samples, num_classes), device=device) / num_classes
        uncond = one_hot.clone()
    elif class_label is None and num_classes > 0:
        # Random class
        random_labels = torch.randint(0, num_classes, (num_samples,), device=device)
        one_hot = torch.eye(num_classes, device=device)[random_labels]
        uncond = torch.zeros_like(one_hot)
    elif class_label is not None:
        # Specific class
        random_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
        one_hot = torch.eye(num_classes, device=device)[random_labels]
        uncond = torch.zeros_like(one_hot)
    else:
        one_hot = None
        uncond = None

    return one_hot, uncond, random_labels


@torch.no_grad()
def generate_with_mask_multistep(
    adapter: GeneratorAdapter,
    masker: Optional[ActivationMasker] = None,
    class_label: Optional[int] = None,
    caption: Optional[str] = None,
    num_steps: int = 4,
    mask_steps: Optional[int] = None,
    noise_level_max: float = 100.0,
    noise_level_min: float = 0.0,
    rho: float = 7.0,
    guidance_scale: float = 1.0,
    stochastic: bool = True,  # Deprecated, ignored
    noise_mode: str = "stochastic",  # Only stochastic supported; see Phase 5
    num_samples: int = 1,
    device: str = 'cuda',
    seed: Optional[int] = None,
    extract_layers: Optional[List[str]] = None,
    return_trajectory: bool = False,
    return_intermediates: bool = False,
    return_noised_inputs: bool = False
):
    """
    Generate images using multi-step denoising with optional activation masking.

    Args:
        adapter: GeneratorAdapter instance
        masker: ActivationMasker with masks set (hooks should be registered)
        class_label: Class label (0-999), random if None, -1 for uniform
        caption: Text caption for T2I models (encoded by adapter), overrides class_label
        num_steps: Number of denoising steps
        mask_steps: Steps to apply mask (default=num_steps, 1=first-only)
        noise_level_max: Starting noise level (0-100), default 100 (pure noise)
        noise_level_min: Ending noise level (0-100), default 0 (clean)
        rho: Karras schedule parameter (for EDM-style models)
        guidance_scale: CFG scale (0=uncond, 1=class, >1=amplify)
        stochastic: Legacy param, ignored when noise_mode is set
        noise_mode: Noise injection mode:
            "stochastic" - fresh random noise each step (default)
            "fixed" - pre-generated noise reused across generations (seeded)
            "zero" - no noise (deterministic, zeros)
        num_samples: Number of images
        device: Device for generation
        seed: Random seed
        extract_layers: Layers to extract for trajectory
        return_trajectory: Return activations at each step
        return_intermediates: Return intermediate images
        return_noised_inputs: Return noised input x_t at each step

    Returns:
        (images, labels) or (images, labels, trajectory) or
        (images, labels, trajectory, intermediates) or
        (images, labels, trajectory, intermediates, noised_inputs)
    """
    if mask_steps is None:
        mask_steps = num_steps
    if seed is not None or noise_mode in ("zero", "fixed"):
        torch.manual_seed(seed if seed is not None else 42)
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Get timesteps/sigmas from adapter using universal noise_level (0-100)
    timesteps = adapter.get_timesteps(
        num_steps,
        device=device,
        noise_level_max=noise_level_max,
        noise_level_min=noise_level_min,
        rho=rho
    )

    # Prepare conditioning using helper
    cond, uncond, random_labels = _prepare_conditioning(
        adapter, class_label, caption, num_samples, device
    )

    # Setup trajectory extraction (diffviews-specific)
    trajectory_activations = []
    intermediate_images = []
    noised_input_images = []
    extractor = None

    if return_trajectory and extract_layers:
        extractor = ActivationExtractor(adapter, extract_layers)
        extractor.register_hooks()

    # Generate initial noise using adapter (handles model-specific scaling)
    x = adapter.get_initial_noise(
        batch_size=num_samples,
        device=device,
        noise_level=noise_level_max
    )

    # Iterative denoising
    for i, t in enumerate(timesteps[:-1]):
        # Capture noised input before denoising (decode for latent models)
        if return_noised_inputs:
            noised_input_images.append(tensor_to_uint8_image(adapter.decode(x)))

        # Remove mask after mask_steps
        if i == mask_steps and masker is not None:
            masker.remove_hooks()

        # Forward with CFG using adapter interface
        pred = adapter.forward_with_cfg(x, t, cond, uncond, guidance_scale)

        # Extract trajectory activations
        if extractor is not None:
            acts = extractor.get_activations()
            layer_acts = []
            for layer_name in sorted(extract_layers):
                act = acts.get(layer_name)
                if act is not None:
                    if len(act.shape) == 4:
                        B, C, H, W = act.shape
                        act = act.reshape(B, -1)
                    layer_acts.append(act.numpy())
            if layer_acts:
                concat_act = np.concatenate(layer_acts, axis=1)
                trajectory_activations.append(concat_act)
            extractor.clear()

        # Capture intermediate image (decode for latent models)
        if return_intermediates:
            # Convert prediction to estimated x0 (handles epsilon/sample/v_prediction)
            x0_estimate = adapter.pred_to_sample(x, t, pred)
            intermediate_images.append(tensor_to_uint8_image(adapter.decode(x0_estimate)))

        # Model-appropriate step via adapter interface
        t_next = timesteps[i + 1]
        x = adapter.step(x, t, pred, t_next=t_next)

    if extractor is not None:
        extractor.remove_hooks()

    # Decode latents to pixel space (identity for pixel-space models)
    x = adapter.decode(x)

    # Convert to uint8
    images = tensor_to_uint8_image(x)

    # Build return tuple
    result = [images, random_labels.cpu()]
    if return_trajectory:
        result.append(trajectory_activations)
    if return_intermediates:
        result.append(intermediate_images)
    if return_noised_inputs:
        result.append(noised_input_images)

    return tuple(result) if len(result) > 2 else (result[0], result[1])


def save_generated_sample(
    image: torch.Tensor,
    activations: Dict,
    metadata: Dict,
    output_dir: Path,
    sample_id: str
) -> Dict:
    """
    Save generated image, activations, and metadata.

    Args:
        image: (H, W, 3) uint8 tensor
        activations: Dict of layer_name -> activation tensor
        metadata: Dict with sample info
        output_dir: Root output directory
        sample_id: Unique sample identifier

    Returns:
        Record dict with paths and metadata
    """
    output_dir = Path(output_dir)
    model_name = metadata.get("model", "generated")

    # Save image
    image_dir = output_dir / "images" / model_name
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{sample_id}.png"

    image_pil = Image.fromarray(image.numpy())
    image_pil.save(image_path)

    # Save activations if any
    if activations:
        activation_dir = output_dir / "activations" / model_name
        activation_dir.mkdir(parents=True, exist_ok=True)
        activation_path = activation_dir / f"{sample_id}"

        activation_dict = {}
        for name, activation in activations.items():
            if isinstance(activation, torch.Tensor):
                if len(activation.shape) == 4:
                    B, C, H, W = activation.shape
                    activation_dict[name] = activation.reshape(B, -1).cpu().numpy()
                else:
                    activation_dict[name] = activation.cpu().numpy()
            else:
                activation_dict[name] = activation

        np.savez_compressed(str(activation_path.with_suffix('.npz')), **activation_dict)

        with open(activation_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    return {
        'sample_id': sample_id,
        'image_path': f"images/{model_name}/{sample_id}.png",
        **metadata
    }


def infer_layer_shape(
    adapter: GeneratorAdapter, layer_name: str, device: str = 'cuda'
) -> Tuple[int, ...]:
    """
    Infer activation shape for a layer by running dummy forward pass.

    Args:
        adapter: GeneratorAdapter instance
        layer_name: Layer to infer shape for
        device: Device for inference

    Returns:
        (C, H, W) shape tuple
    """
    shapes = adapter.get_layer_shapes()
    if layer_name in shapes:
        return shapes[layer_name]

    # Fallback: run dummy inference
    resolution = adapter.resolution
    num_classes = adapter.num_classes

    dummy_noise = torch.randn(1, 3, resolution, resolution, device=device)
    dummy_label = torch.zeros(1, num_classes, device=device)
    dummy_label[0, 0] = 1.0
    dummy_sigma = torch.ones(1, device=device) * 80.0

    extractor = ActivationExtractor(adapter, [layer_name])
    extractor.register_hooks()

    adapter.forward(dummy_noise * 80.0, dummy_sigma, dummy_label)
    activations = extractor.get_activations()
    extractor.remove_hooks()

    if layer_name not in activations:
        raise ValueError(f"Layer {layer_name} not found during inference")

    shape = activations[layer_name].shape
    return tuple(shape[1:])  # (C, H, W)

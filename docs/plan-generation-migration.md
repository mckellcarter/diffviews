# Migration Plan: Use adapt_diff.generate() in diffviews

**Status: COMPLETE**

## Summary

Update `adapt_diff.generate()` to support `ActivationMasker`, then replace `diffviews/core/generator.py::generate_with_mask_multistep()` with a thin wrapper around it.

---

## Current State

### adapt_diff has:
- `ActivationMasker` in extraction.py (moved from diffviews in previous migration)
- `generate()` in generation.py with sigma_max/sigma_min, caption support, trajectory extraction
- Adapter interface with `step()`, `forward_with_cfg()`, `decode()`

### adapt_diff.generate() is missing:
- `activation_masker` param (to use ActivationMasker)
- `mask_steps` param (when to remove hooks)
- `noise_mode` param ("stochastic", "fixed", "zero")

### diffviews has:
- `generate_with_mask_multistep()` - manual implementation with all features
- Uses ActivationMasker with mask_steps
- Has noise_mode support

### Adapter step() behavior:
- **DMD2:** `return pred + t_next * torch.randn_like(pred)` - adds fresh noise
- **EDM:** Euler step, deterministic (no noise)

---

## Phase 1: Update adapt_diff

### 1.1 Add step_noise param to adapter.step()

**adapt_diff/base.py** - update signature:
```python
def step(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_output: torch.Tensor,
    t_next: Optional[torch.Tensor] = None,
    step_noise: Optional[torch.Tensor] = None,  # NEW
    **kwargs
) -> torch.Tensor:
```

**adapt_diff/adapters/dmd2_imagenet.py**:
```python
def step(self, x_t, t, model_output, t_next=None, step_noise=None, **kwargs):
    if t_next is None or float(t_next) == 0:
        return model_output

    noise = step_noise if step_noise is not None else torch.randn_like(model_output)
    return model_output + t_next * noise
```

**adapt_diff/adapters/edm_imagenet.py** - unchanged (Euler is deterministic):
```python
def step(self, x_t, t, model_output, t_next=None, step_noise=None, **kwargs):
    # step_noise ignored for Euler stepping
    d_cur = (x_t - model_output) / t
    return x_t + (t_next - t) * d_cur
```

### 1.2 Update generate() with ActivationMasker support

**adapt_diff/generation.py** - add params:
```python
from .extraction import ActivationMasker

def generate(
    adapter: GeneratorAdapter,
    class_label: Optional[int] = None,
    caption: Optional[str] = None,
    num_steps: int = 6,
    ...
    # Activation masking
    activation_masker: Optional[ActivationMasker] = None,
    mask_steps: Optional[int] = None,
    # Noise control
    noise_mode: str = "stochastic",
    ...
) -> GenerationResult:
```

**Add noise pre-generation:**
```python
# Pre-generate step noises based on mode
noise_shape = (num_samples, adapter.in_channels, adapter.resolution, adapter.resolution)
if noise_mode == "zero":
    step_noises = [torch.zeros(noise_shape, device=device) for _ in range(num_steps - 1)]
elif noise_mode == "fixed":
    rng = torch.Generator(device=device).manual_seed(seed or 42)
    step_noises = [torch.randn(noise_shape, device=device, generator=rng)
                   for _ in range(num_steps - 1)]
else:  # stochastic
    step_noises = [None] * (num_steps - 1)
```

**Add masker hook management:**
```python
# Register activation masker
if activation_masker is not None:
    activation_masker.register_hooks()
    if mask_steps is None:
        mask_steps = num_steps

try:
    # Denoising loop
    for i, t in enumerate(timesteps[:-1]):
        # Remove masker after mask_steps
        if i == mask_steps and activation_masker is not None:
            activation_masker.remove_hooks()

        pred = adapter.forward_with_cfg(x, t_batched, cond, uncond, guidance_scale)

        # Step with noise control
        t_next = timesteps[i + 1]
        step_noise = step_noises[i] if i < len(step_noises) else None
        x = adapter.step(x, t_batched, pred, t_next=t_next_batched, step_noise=step_noise)

finally:
    # Ensure cleanup
    if activation_masker is not None:
        activation_masker.remove_hooks()
```

### 1.3 Add tests

**adapt_diff/tests/test_generation.py**:
```python
def test_generate_with_activation_masker():
    """ActivationMasker hooks register and cleanup correctly."""
    ...

def test_generate_noise_mode_fixed():
    """Fixed noise mode produces reproducible results."""
    ...

def test_generate_noise_mode_zero():
    """Zero noise mode is deterministic."""
    ...

def test_generate_mask_steps():
    """Masker hooks removed after mask_steps."""
    ...
```

---

## Phase 2: Update diffviews

### 2.1 Replace generator.py implementation

**diffviews/core/generator.py** - thin wrapper:
```python
"""Image generation - wraps adapt_diff.generate()"""

from typing import List, Optional, Tuple
import torch

from adapt_diff import ActivationMasker
from adapt_diff.generation import generate as adapt_generate


def generate_with_mask_multistep(
    adapter,
    masker: Optional[ActivationMasker] = None,
    class_label: Optional[int] = None,
    caption: Optional[str] = None,
    num_steps: int = 4,
    mask_steps: Optional[int] = None,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    rho: float = 7.0,
    guidance_scale: float = 1.0,
    noise_mode: str = "stochastic",
    num_samples: int = 1,
    device: str = 'cuda',
    seed: Optional[int] = None,
    extract_layers: Optional[List[str]] = None,
    return_trajectory: bool = False,
    return_intermediates: bool = False,
    return_noised_inputs: bool = False,
    stochastic: bool = True,  # legacy, ignored
) -> Tuple:
    """Generate images with optional activation masking."""

    result = adapt_generate(
        adapter,
        class_label=class_label,
        caption=caption,
        num_steps=num_steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        guidance_scale=guidance_scale,
        num_samples=num_samples,
        device=device,
        seed=seed,
        activation_masker=masker,
        mask_steps=mask_steps,
        noise_mode=noise_mode,
        extract_layers=extract_layers,
        return_trajectory=return_trajectory,
        return_intermediates=return_intermediates,
        return_noised_inputs=return_noised_inputs,
        rho=rho,
    )

    # Convert to legacy tuple format
    ret = [result.images, result.labels]
    if return_trajectory:
        ret.append(result.trajectory)
    if return_intermediates:
        ret.append(result.intermediates)
    if return_noised_inputs:
        ret.append(result.noised_inputs)

    return tuple(ret) if len(ret) > 2 else (ret[0], ret[1])


def tensor_to_uint8_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] tensor to [0,255] uint8."""
    images = ((tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return images.permute(0, 2, 3, 1).cpu()


# Keep save_generated_sample() and infer_layer_shape() as-is
```

### 2.2 modal_gpu.py - no changes needed

modal_gpu.py already uses `noise_level_max`/`noise_level_min` (the model-agnostic 0-100 scale).
The wrapper accepts both sigma and noise_level params for backwards compatibility.

### 2.3 Update requirements.txt

Pin to new adapt_diff commit after Phase 1 is merged.

---

## Phase 3: Verification

```bash
# adapt_diff
cd /Users/mckell/Documents/GitHub/adapt_diff
pytest tests/ -v

# diffviews
cd /Users/mckell/Documents/GitHub/diffviews
pytest tests/test_generator.py tests/test_masking.py -v

# Modal
modal run modal_gpu.py
```

---

## Files Modified

**adapt_diff:**
- `adapt_diff/base.py` - add step_noise to step() signature
- `adapt_diff/adapters/dmd2_imagenet.py` - implement step_noise
- `adapt_diff/adapters/edm_imagenet.py` - add step_noise param (ignored)
- `adapt_diff/adapters/mscoco_t2i.py` - add step_noise param if needed
- `adapt_diff/generation.py` - add activation_masker, mask_steps, noise_mode
- `tests/test_generation.py` - new tests

**diffviews:**
- `diffviews/core/generator.py` - replaced with wrapper (accepts both sigma and noise_level)
- `requirements.txt` - updated adapt_diff hash to 017c923

---

## Execution Order

1. Update adapt_diff (Phase 1)
2. Run adapt_diff tests
3. Commit adapt_diff, get hash
4. Update diffviews (Phase 2)
5. Update requirements.txt with new hash
6. Run diffviews tests (Phase 3)
7. Test modal deployment

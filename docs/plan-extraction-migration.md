# Migration Plan: Consolidate Extraction/Masking Code

**Status: COMPLETED**

- adapt_diff: `feature/extraction-utilities` branch, commit `c943aa1`
- diffviews: `feature/extraction-migration` branch, commit `941c20f`

## Summary

Move `ActivationMasker` and related utilities from diffviews to adapt_diff. Delete duplicate extraction code from diffviews and re-export from adapt_diff.

## What Moves Where

| Code | From | To | Action |
|------|------|-----|--------|
| `ActivationMasker` | diffviews/core/masking.py | adapt_diff/extraction.py | MOVE |
| `unflatten_activation()` | diffviews/core/masking.py | adapt_diff/extraction.py | MOVE |
| `load_activation_from_npz()` | diffviews/core/masking.py | adapt_diff/extraction.py | MOVE |
| `ActivationExtractor` | diffviews/core/extractor.py | (already in adapt_diff) | DELETE diffviews copy |
| `flatten_activations()` | diffviews/core/extractor.py | (already in adapt_diff) | DELETE diffviews copy |
| `load_activations()` | diffviews/core/extractor.py | (already in adapt_diff) | DELETE diffviews copy |
| `convert_to_fast_format()` | diffviews/core/extractor.py | (already in adapt_diff) | DELETE diffviews copy |
| `load_fast_activations()` | diffviews/core/extractor.py | (already in adapt_diff) | DELETE diffviews copy |
| `compute_mask_dict()` | diffviews/core/masking.py | (stays) | KEEP (UMAP-specific) |

---

## Phase 1: adapt_diff Changes

### 1.1 Add to `adapt_diff/extraction.py`

Add torch import and new code after existing functions (~line 227):

```python
import torch  # add to imports

class ActivationMasker:
    """Mask layer activations with fixed values during forward pass."""

    def __init__(self, adapter: "GeneratorAdapter"):
        self.adapter = adapter
        self.masks: Dict[str, torch.Tensor] = {}
        self._handles = []

    def set_mask(self, layer_name: str, activation: torch.Tensor):
        self.masks[layer_name] = activation.cpu()

    def clear_mask(self, layer_name: str):
        self.masks.pop(layer_name, None)

    def clear_masks(self):
        self.masks.clear()

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if name not in self.masks:
                return output
            mask = self.masks[name]
            if isinstance(output, tuple):
                target = output[0]
                masked = mask.to(target.device, target.dtype)
                if masked.shape[0] == 1 and target.shape[0] > 1:
                    masked = masked.expand(target.shape[0], -1, -1, -1)
                return (masked,) + output[1:]
            else:
                masked = mask.to(output.device, output.dtype)
                if masked.shape[0] == 1 and output.shape[0] > 1:
                    masked = masked.expand(output.shape[0], -1, -1, -1)
                return masked
        return hook

    def register_hooks(self, layers: List[str] = None):
        if layers is None:
            layers = list(self.masks.keys())
        for name in layers:
            hook_fn = self._make_hook(name)
            handles = self.adapter.register_activation_hooks([name], hook_fn)
            self._handles.extend(handles)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def unflatten_activation(flat_activation: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Reshape (1, C*H*W) -> (1, C, H, W)."""
    if len(flat_activation.shape) == 1:
        flat_activation = flat_activation.unsqueeze(0)
    B = flat_activation.shape[0]
    C, H, W = target_shape
    return flat_activation.reshape(B, C, H, W)


def load_activation_from_npz(npz_path: Path, layer_name: str) -> torch.Tensor:
    """Load single layer from NPZ file."""
    data = np.load(str(npz_path))
    if layer_name not in data:
        raise ValueError(f"Layer '{layer_name}' not found. Available: {list(data.keys())}")
    activation = torch.from_numpy(data[layer_name])
    if len(activation.shape) == 1:
        activation = activation.unsqueeze(0)
    return activation
```

### 1.2 Update `adapt_diff/extraction.py` `__all__`

```python
__all__ = [
    "ActivationExtractor",
    "ActivationMasker",
    "flatten_activations",
    "load_activations",
    "save_activations",
    "convert_to_fast_format",
    "load_fast_activations",
    "get_fast_format_info",
    "unflatten_activation",
    "load_activation_from_npz",
]
```

### 1.3 Update `adapt_diff/__init__.py`

Add imports:
```python
from .extraction import (
    flatten_activations,
    load_activations,
    save_activations,
    convert_to_fast_format,
    load_fast_activations,
    get_fast_format_info,
    ActivationMasker,
    unflatten_activation,
    load_activation_from_npz,
)
```

Add to `__all__`:
```python
    "ActivationMasker",
    "unflatten_activation",
    "load_activation_from_npz",
```

### 1.4 Add tests: `adapt_diff/tests/test_masking.py`

Test ActivationMasker, unflatten_activation, load_activation_from_npz with MockAdapter.

---

## Phase 2: diffviews Changes

### 2.1 DELETE `diffviews/core/extractor.py`

Entire file removed.

### 2.2 UPDATE `diffviews/core/masking.py`

Keep ONLY `compute_mask_dict()`. Remove ActivationMasker, unflatten_activation, load_activation_from_npz.

### 2.3 UPDATE `diffviews/core/__init__.py`

```python
"""Core activation extraction and masking functionality."""

from adapt_diff import (
    ActivationExtractor,
    ActivationMasker,
    flatten_activations,
    load_activations,
    convert_to_fast_format,
    load_fast_activations,
    unflatten_activation,
    load_activation_from_npz,
)
from .masking import compute_mask_dict

__all__ = [
    "ActivationExtractor",
    "ActivationMasker",
    "flatten_activations",
    "load_activations",
    "convert_to_fast_format",
    "load_fast_activations",
    "unflatten_activation",
    "load_activation_from_npz",
    "compute_mask_dict",
]
```

### 2.4 Update imports in diffviews files

| File | Change |
|------|--------|
| `diffviews/core/generator.py:14-15` | `from adapt_diff import ActivationExtractor, ActivationMasker` |
| `diffviews/visualization/gpu_ops.py:10` | `from adapt_diff import ActivationMasker` + `from diffviews.core.masking import compute_mask_dict` |
| `diffviews/visualization/visualizer.py:30` | `from adapt_diff import unflatten_activation` |
| `diffviews/processing/umap.py:17` | `from adapt_diff import flatten_activations, load_activations, load_fast_activations` |
| `scripts/extract_mscoco_activations.py:28` | `from adapt_diff import ActivationExtractor` |
| `diffviews/scripts/cli.py:108` | `from adapt_diff import convert_to_fast_format` |
| `scripts/create_demo_subset.py:14` | `from adapt_diff import load_activations, flatten_activations` |

### 2.5 Update test imports

| File | Change |
|------|--------|
| `tests/test_masking.py:12-16` | `from adapt_diff import ActivationMasker, load_activation_from_npz, unflatten_activation` |
| `tests/test_generator.py:19` | `from adapt_diff import ActivationMasker` |

---

## Execution Order

1. **adapt_diff changes first** (Phase 1)
2. **Commit and get hash**
3. **Update diffviews requirements.txt** to pin new adapt_diff commit
4. **diffviews changes** (Phase 2)

---

## Verification

```bash
# adapt_diff
cd /Users/mckell/Documents/GitHub/adapt_diff
pytest tests/ -v

# diffviews
cd /Users/mckell/Documents/GitHub/diffviews
pytest tests/ -v

# API check
python -c "from diffviews.core import ActivationMasker, compute_mask_dict; print('OK')"
python -c "from adapt_diff import ActivationMasker, unflatten_activation; print('OK')"
```

---

## Files Modified

**adapt_diff:**
- `adapt_diff/extraction.py` - add ActivationMasker, unflatten_activation, load_activation_from_npz
- `adapt_diff/__init__.py` - export new symbols
- `adapt_diff/tests/test_masking.py` - new test file

**diffviews:**
- `diffviews/core/extractor.py` - DELETE
- `diffviews/core/masking.py` - keep only compute_mask_dict
- `diffviews/core/__init__.py` - re-export from adapt_diff
- `diffviews/core/generator.py` - update imports
- `diffviews/visualization/gpu_ops.py` - update imports
- `diffviews/visualization/visualizer.py` - update imports
- `diffviews/processing/umap.py` - update imports
- `diffviews/scripts/cli.py` - update imports
- `scripts/extract_mscoco_activations.py` - update imports
- `scripts/create_demo_subset.py` - update imports
- `tests/test_masking.py` - update imports
- `tests/test_generator.py` - update imports

---

## Results

**Tests passing:**
- adapt_diff: 34/34 tests pass
- diffviews: 49/54 tests pass (5 pre-existing failures unrelated to migration)

**API verification:**
```bash
python -c "from diffviews.core import ActivationMasker, compute_mask_dict; print('OK')"  # OK
python -c "from adapt_diff import ActivationMasker, unflatten_activation; print('OK')"   # OK
```

**Lines changed:**
- adapt_diff: +494 lines (new masking code + tests)
- diffviews: -446 lines, +121 lines (net reduction of ~325 lines)

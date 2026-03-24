# Plan: Replace diffviews/adapters with adapt_diff

## Summary
Strip the `diffviews/adapters/` module and `diffviews/vendor/` directory from this repo, replacing all adapter imports with the external `adapt_diff` package at `https://github.com/mckellcarter/adapt_diff`.

**Approach:** Clean break - no backwards compatibility shims.

---

## Step 0: Git Setup

```bash
# Stash current README changes
git stash push -m "README doc fixes"

# Fetch and checkout main
git fetch origin
git checkout main
git pull origin main

# Create new feature branch
git checkout -b feature/use-adapt-diff

# Apply stashed README changes
git stash pop
```

## API Compatibility
adapt_diff is **99% compatible** with diffviews/adapters:
- `GeneratorAdapter` base class: identical
- `HookMixin`: identical
- Registry functions (`get_adapter`, `list_adapters`, `register_adapter`): identical
- Only difference: entry points group name (`adapt_diff.adapters` vs `diffviews.adapters`)

---

## Migration Steps

### Step 1: Add adapt_diff dependency

**pyproject.toml** - add to dependencies array:
```toml
"adapt_diff @ git+https://github.com/mckellcarter/adapt_diff.git@main",
```

**requirements.txt** - add line:
```
adapt_diff @ git+https://github.com/mckellcarter/adapt_diff.git@main
```

### Step 2: Update core module imports

| File | Line | Change |
|------|------|--------|
| `diffviews/core/extractor.py` | 12 | `from adapt_diff import GeneratorAdapter` |
| `diffviews/core/masking.py` | 10 | `from adapt_diff import GeneratorAdapter` |
| `diffviews/core/generator.py` | 13 | `from adapt_diff import GeneratorAdapter` |
| `diffviews/visualization/visualizer.py` | 29 | `from adapt_diff import get_adapter` |
| `modal_gpu.py` | 75 | `from adapt_diff import get_adapter` |

### Step 3: Update test imports

| File | Line | Change |
|------|------|--------|
| `tests/test_generator.py` | 21 | `from adapt_diff import GeneratorAdapter` |
| `tests/test_masking.py` | 17 | `from adapt_diff import GeneratorAdapter` |
| `tests/test_dmd2_adapter.py` | 6 | `from adapt_diff import get_adapter, list_adapters` |

### Step 4: Update diffviews/__init__.py

Remove all adapter exports (clean break):
```python
# Remove these lines:
# from diffviews.adapters.base import GeneratorAdapter
# from diffviews.adapters.registry import get_adapter, list_adapters, register_adapter
```

### Step 5: Remove entry points from pyproject.toml

Delete these lines:
```toml
[project.entry-points."diffviews.adapters"]
# Adapters register themselves here
# Example: imagenet-64 = "diffviews_dmd2:DMD2ImageNetAdapter"
```

### Step 6: Delete adapter module and vendor directories

```bash
rm -rf diffviews/adapters/
rm -rf diffviews/vendor/
```

Files deleted:
- `diffviews/adapters/__init__.py`
- `diffviews/adapters/base.py`
- `diffviews/adapters/dmd2_imagenet.py`
- `diffviews/adapters/edm_imagenet.py`
- `diffviews/adapters/hooks.py`
- `diffviews/adapters/registry.py`
- `diffviews/adapters/nvidia_compat.py`
- `diffviews/vendor/dnnlib/` (all files)
- `diffviews/vendor/torch_utils/` (all files)

### Step 7: Update docs/ARCHITECTURE.md

Replace Adapters Module section with brief note pointing to adapt_diff repo.

### Step 8: Run tests and linter

```bash
pytest tests/
pylint diffviews/
```

---

## Verification

```bash
# Install updated deps
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
pylint diffviews/

# Test local app (if checkpoints available)
python -m diffviews.visualization.app
```

---

## Rollback

If issues arise, revert the commit. The adapters module is self-contained so no partial state to worry about.

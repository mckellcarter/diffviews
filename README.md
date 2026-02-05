# DiffViews

Model-agnostic diffusion activation visualizer.

## Overview

DiffViews provides tools for visualizing and exploring the internal activations of diffusion models through UMAP embeddings. It uses an adapter interface to support different model architectures.

<img width="1943" height="1284" alt="image" src="https://github.com/user-attachments/assets/e3861312-e7a8-48ca-a37b-7278f011d287" />

## Features

- **Model-agnostic**: Works with any diffusion model via the `GeneratorAdapter` interface
- **Activation extraction**: Extract and save layer activations during generation
- **Activation masking**: Constrain generation by fixing specific layer outputs
- **UMAP visualization**: Project high-dimensional activations to 2D/3D
- **Interactive dashboard**: Gradio app to explore embeddings, find neighbors, generate from activations
- **Multi-step generation**: Support for single-step and multi-step denoising with CFG

## Installation

```bash
pip install diffviews

# With visualization dashboard
pip install diffviews[viz]

# With LMDB support
pip install diffviews[lmdb]

# All extras
pip install diffviews[all]
```

## Quick Start

### Using with a model adapter

```python
from diffviews import get_adapter
from diffviews.core import ActivationExtractor, generate_with_mask_multistep

# Load adapter (registered via entry point)
AdapterClass = get_adapter('imagenet-64')
adapter = AdapterClass.from_checkpoint('path/to/model.pth', device='cuda')

# Extract activations
extractor = ActivationExtractor(adapter, layers=['encoder_bottleneck', 'midblock'])
with extractor:
    images, labels = adapter.forward(noise, sigma, class_labels)
    activations = extractor.get_activations()

# Generate with masked activations
from diffviews.core import ActivationMasker
masker = ActivationMasker(adapter)
masker.set_mask('encoder_bottleneck', target_activation)
masker.register_hooks()

images, labels = generate_with_mask_multistep(
    adapter, masker,
    class_label=207,  # golden retriever
    num_steps=4,
    guidance_scale=1.5
)
```

### Creating a custom adapter

```python
from diffviews.adapters import GeneratorAdapter, HookMixin, register_adapter

@register_adapter('my-model')
class MyModelAdapter(HookMixin, GeneratorAdapter):
    def __init__(self, model, device):
        HookMixin.__init__(self)
        self._model = model
        self._device = device

    @property
    def model_type(self): return 'my-model'

    @property
    def resolution(self): return 256

    @property
    def num_classes(self): return 1000

    @property
    def hookable_layers(self):
        return ['encoder', 'decoder', 'mid']

    def forward(self, x, sigma, class_labels=None, **kwargs):
        return self._model(x, sigma, class_labels)

    def register_activation_hooks(self, layer_names, hook_fn):
        handles = []
        for name in layer_names:
            module = self._get_module(name)
            h = module.register_forward_hook(hook_fn)
            handles.append(h)
            self.add_handle(h)
        return handles

    def get_layer_shapes(self):
        # Return cached or computed shapes
        return {'encoder': (512, 16, 16), 'decoder': (256, 32, 32), 'mid': (512, 8, 8)}

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda', **kwargs):
        model = load_my_model(checkpoint_path)
        model = model.to(device).eval()
        return cls(model, device)

    @classmethod
    def get_default_config(cls):
        return {'resolution': 256, 'channels': 3}
```

## Supported Models

DiffViews supports multiple diffusion models via the adapter interface:

| Model | Adapter | Steps | Description |
|-------|---------|-------|-------------|
| DMD2 | `dmd2-imagenet-64` | 1-10 | Distribution Matching Distillation (single/few-step) |
| EDM | `edm-imagenet-64` | 50-256 | Elucidating Diffusion Models (multi-step) |

## Setup

### 1. Install

```bash
pip install diffviews[viz]
```

### 2. Download Data & Checkpoints

Data hosted on Cloudflare R2 (HuggingFace fallback). One command downloads everything:

```bash
# Downloads data (~1.7GB) + checkpoints (~2.4GB)
diffviews download
```

> **Security Warning**: Model checkpoints are pickle (`.pkl`) files which can execute arbitrary code. The checkpoints downloaded by this script are from official sources (NVIDIA EDM, converted DMD2).

**Options:**
```bash
# Skip checkpoints (visualization only, no generation)
diffviews download --checkpoints none

# Download only specific checkpoint
diffviews download --checkpoints dmd2
diffviews download --checkpoints edm

# Force data source
diffviews download --source r2    # Cloudflare R2 only
diffviews download --source hf    # HuggingFace only
```

### 3. Run

```bash
diffviews viz --data-dir data
```

Or without install: `python -m diffviews.scripts.cli viz --data-dir data`

Device is auto-detected (CUDA > MPS > CPU). Override with `--device cuda|mps|cpu`.

### Data Directory Structure

After download, the `data/` directory contains:

```
data/
├── dmd2/
│   ├── config.json           # Model config (adapter, defaults)
│   ├── checkpoints/          # Model weights (.pkl)
│   ├── embeddings/           # UMAP coordinates (.csv) + model (.pkl)
│   ├── activations/          # Pre-concatenated activations (.npy)
│   ├── images/               # Source images
│   └── metadata/             # Class labels
└── edm/
    └── (same structure)
```

## Deployment

### Modal (serverless GPU)

```bash
pip install modal
modal serve modal_app.py   # dev
modal deploy modal_app.py  # prod
```

Single A10G container, data from R2, scales to zero when idle.

### Local

```bash
pip install diffviews[viz]
diffviews download
diffviews viz-gradio --data-dir data
```

### Gradio Features

- **Model Switching**: Dropdown to switch between discovered models
- **Point Selection**: Click points to select, click neighbors to add/remove
- **KNN Suggestions**: Auto-suggest nearest neighbors with distance display
- **Generation**: Generate from averaged neighbor activations
- **Trajectory View**: Denoising path visualized on UMAP with sigma labels
- **Intermediate Steps**: Gallery of denoising steps with frame navigation
- **Hover Preview**: Preview samples by hovering over points

### CLI Options

```bash
diffviews viz-gradio --help
```

| Option | Description |
|--------|-------------|
| `--data-dir` | Root data directory (parent with model subdirs, or single model dir) |
| `--embeddings` | Pre-computed UMAP embeddings CSV (optional in multi-model mode) |
| `--checkpoint-path` | Path to model checkpoint pkl (optional if in config.json) |
| `--adapter` | Adapter name: `dmd2-imagenet-64`, `edm-imagenet-64` |
| `--device` | `cuda`, `mps`, or `cpu` (auto-detected if omitted) |
| `--num-steps` | Denoising steps (overrides config default) |
| `--guidance-scale` | CFG scale (0=uncond, 1=class-cond, >1=amplified) |
| `--model`, `-m` | Initial model to load (e.g., `dmd2`, `edm`) |
| `--port` | Server port (default: 7860) |
| `--share` | Create public share link (Gradio only) |

### Other Commands

```bash
# Convert .npz activations to fast .npy format (if you have old data)
diffviews convert data/dmd2

# Show all commands
diffviews --help
```

## Package Structure

```
diffviews/
├── adapters/           # Model adapter interface
│   ├── base.py         # GeneratorAdapter ABC
│   ├── hooks.py        # HookMixin utilities
│   └── registry.py     # Adapter registration
├── core/               # Core functionality
│   ├── extractor.py    # Activation extraction + fast format conversion
│   ├── masking.py      # Activation masking
│   └── generator.py    # Generation utilities
├── processing/         # UMAP computation
│   └── umap.py
├── scripts/            # CLI entry points
│   └── cli.py          # diffviews command (convert, viz)
├── data/               # R2 data cache + store
├── visualization/      # Gradio interactive app
└── utils/              # Utilities
    ├── device.py
    └── checkpoint.py
```

## Adapter Registration

Adapters can register via Python entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."diffviews.adapters"]
imagenet-64 = "my_package.adapters:ImageNetAdapter"
sdxl = "my_package.adapters:SDXLAdapter"
```

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

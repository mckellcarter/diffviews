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
import torch
from adapt_diff import get_adapter
from diffviews.core.extractor import ActivationExtractor
from diffviews.core.generator import generate_with_mask_multistep

# Load adapter
AdapterClass = get_adapter('dmd2-imagenet-64')
adapter = AdapterClass.from_checkpoint('path/to/model.pkl', device='cuda')

# Get defaults from adapter config
config = adapter.get_default_config()
# config contains: noise_max (100), noise_min (0-5), default_steps, etc.

# Extract activations
extractor = ActivationExtractor(adapter, layers=['encoder_bottleneck', 'midblock'])
with extractor:
    timestep = adapter.noise_level_to_native(torch.tensor(80.0))  # 80% noise
    images, labels = adapter.forward(noise, timestep, class_labels)
    activations = extractor.get_activations()

# Generate with masked activations
from diffviews.core.masking import ActivationMasker
masker = ActivationMasker(adapter)
masker.set_mask('encoder_bottleneck', target_activation)
masker.register_hooks()

images, labels = generate_with_mask_multistep(
    adapter, masker,
    class_label=207,  # golden retriever
    num_steps=4,
    noise_level_max=100.0,  # start from pure noise (0-100 scale)
    noise_level_min=0.0,    # denoise to clean
    guidance_scale=1.5
)
```

### Creating a custom adapter

Adapters are defined in the [`adapt_diff`](https://github.com/mckellcarter/adapt_diff) package. See that repository for creating new adapters.

```python
from adapt_diff import GeneratorAdapter, HookMixin, register_adapter

@register_adapter('my-model')
class MyModelAdapter(HookMixin, GeneratorAdapter):
    # See adapt_diff documentation for full implementation details
    ...
```

## Supported Models

DiffViews supports multiple diffusion models via adapters from [`adapt_diff`](https://github.com/mckellcarter/adapt_diff):

| Model | Adapter | Steps | Description |
|-------|---------|-------|-------------|
| DMD2 | `dmd2-imagenet-64` | 1-10 | Distribution Matching Distillation (single/few-step) |
| EDM | `edm-imagenet-64` | 50-256 | Elucidating Diffusion Models (multi-step) |
| MSCOCO T2I | `mscoco-t2i-128` | — | Text-to-Image 128x128 |
| Custom SD | `abu-custom-sd14` | — | Custom Stable Diffusion v1.4 |

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

**Hybrid architecture (recommended):** CPU container serves UI, GPU container handles generation.

```bash
pip install modal
modal deploy modal_gpu.py  # GPU worker first
modal deploy modal_web.py  # CPU web server
```

- CPU container: Gradio UI, visualization, 30 min scaledown
- GPU container: Lightweight (torch only), 5-10s cold start, T4 GPU
- Cost optimized: GPU only spins up for generation

**Monolithic (simpler, for dev):**

```bash
modal serve modal_app.py   # dev
modal deploy modal_app.py  # prod
```

### Local - requires umaps be fitted locally

```bash
pip install diffviews[viz]
diffviews download
diffviews viz --data-dir data
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
    └── device.py
```

Adapters are provided by the external [`adapt_diff`](https://github.com/mckellcarter/adapt_diff) package.

## Adapter Registration

Adapters register via the `adapt_diff` entry points system:

```toml
# In your package's pyproject.toml
[project.entry-points."adapt_diff.adapters"]
my-model = "my_package.adapters:MyModelAdapter"
```

See the [adapt_diff documentation](https://github.com/mckellcarter/adapt_diff) for details.

## Known Issues

### Noise modes (fixed/zero)

Only `stochastic` noise mode currently works. The `fixed` and `zero` modes require stochastic step support in adapt_diff (planned).

### Multi-tab session corruption (Modal deployment)

When running multiple browser tabs against the same Modal deployment, one tab's Gradio session can become stale or corrupted. Symptoms include:
- Hover preview stuck on a single image
- Point selection not responding

**Workaround**: Switch to a different model in the affected tab, then switch back. This forces a full state reset and restores functionality.

This appears to be a Gradio session edge case rather than a backend state issue. Low priority unless it becomes frequent.

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

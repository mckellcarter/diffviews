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
- **Interactive dashboard**: Explore embeddings, find neighbors, generate from activations
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

## Demo

Quick start with included demo data (16 ImageNet classes, ~1168 samples):

```bash
# Install with visualization extras
pip install diffviews[viz]

# Download checkpoint from HuggingFace (~3.4GB)
python scripts/download_checkpoint.py --output_dir checkpoints/dmd2

# Run demo (visualization only)
python -m diffviews.visualization.app \
  --data_dir demo_data \
  --embeddings demo_data/embeddings/demo_embeddings.csv \
  --device mps  # or cuda/cpu

# Run demo with generation
python -m diffviews.visualization.app \
  --data_dir demo_data \
  --embeddings demo_data/embeddings/demo_embeddings.csv \
  --checkpoint_path checkpoints/dmd2 \
  --adapter dmd2-imagenet-64 \
  --num_steps 6 --mask_steps 1 \
  --guidance_scale 1.0 --label_dropout 0.1 \
  --device mps
```

## Visualization App

Launch the interactive dashboard:

```bash
python -m diffviews.visualization.app \
  --adapter dmd2-imagenet-64 \
  --checkpoint_path /path/to/checkpoint/ \
  --embeddings /path/to/umap_embeddings.csv \
  --data_dir /path/to/image_data/ \
  --device cuda \
  --port 8050
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--adapter` | Registered adapter name (e.g., `dmd2-imagenet-64`) |
| `--checkpoint_path` | Path to model checkpoint file/directory |
| `--embeddings` | Pre-computed UMAP embeddings CSV |
| `--data_dir` | Directory containing image data (NPZ/JPEG/LMDB) |
| `--device` | `cuda`, `mps`, or `cpu` |
| `--num_steps` | Denoising steps (1=single-step, 4/10=multi-step) |
| `--guidance_scale` | CFG scale (0=uncond, 1=class-cond, >1=amplified) |
| `--label_dropout` | Use 0.1 for CFG-trained models |
| `--umap_n_neighbors` | UMAP n_neighbors parameter |
| `--umap_min_dist` | UMAP min_dist parameter |

## Package Structure

```
diffviews/
├── adapters/           # Model adapter interface
│   ├── base.py         # GeneratorAdapter ABC
│   ├── hooks.py        # HookMixin utilities
│   └── registry.py     # Adapter registration
├── core/               # Core functionality
│   ├── extractor.py    # Activation extraction
│   ├── masking.py      # Activation masking
│   └── generator.py    # Generation utilities
├── processing/         # UMAP computation
│   └── umap.py
├── data/               # Data loading
│   ├── sources.py      # NPZ/JPEG/LMDB sources
│   └── class_labels.py
├── visualization/      # Dash interactive app
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

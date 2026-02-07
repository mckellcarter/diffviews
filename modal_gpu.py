"""
Modal GPU worker for diffviews — lightweight generation engine.

Receives pre-computed masks from CPU container, runs generation only.
Minimal dependencies for fast cold start.

Usage:
    modal deploy modal_gpu.py
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal

app = modal.App("diffviews-gpu")

# Minimal GPU image — only what's needed for generation
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "tqdm>=4.60.0",
    )
    # TODO: revert to @main before merging
    .pip_install("diffviews @ git+https://github.com/mckellcarter/diffviews.git@feature/modal-migrate")
)

# Volume for checkpoints only
vol = modal.Volume.from_name("diffviews-data", create_if_missing=True)

DATA_DIR = Path("/data")


def _get_device() -> str:
    """Detect GPU device."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=300,
    scaledown_window=60,
)
class GPUWorker:
    """Lightweight GPU worker — loads adapter, runs generation from pre-computed masks."""

    @modal.enter()
    def setup(self):
        """Called once when container starts."""
        print("=" * 50)
        print("DiffViews GPU Worker (lightweight) starting...")
        print("=" * 50)

        self.device = _get_device()
        print(f"  Device: {self.device}")

        # Cache for loaded adapters
        self._adapters: Dict[str, any] = {}
        self._layer_shapes: Dict[str, Dict[str, Tuple]] = {}

        print("GPU Worker ready.")

    def _get_or_load_adapter(self, model_name: str):
        """Load adapter if not cached."""
        if model_name in self._adapters:
            return self._adapters[model_name]

        from diffviews.adapters import get_adapter

        # Find checkpoint
        checkpoint_dir = DATA_DIR / model_name / "checkpoints"
        if not checkpoint_dir.exists():
            print(f"[GPU] No checkpoint dir for {model_name}")
            return None

        checkpoints = list(checkpoint_dir.glob("*.pkl"))
        if not checkpoints:
            print(f"[GPU] No checkpoint files for {model_name}")
            return None

        checkpoint_path = checkpoints[0]
        print(f"[GPU] Loading adapter from {checkpoint_path}")

        # Determine adapter type from model name
        adapter_type = "dmd2-imagenet-64" if "dmd2" in model_name else "edm-imagenet-64"
        AdapterClass = get_adapter(adapter_type)

        adapter = AdapterClass.from_checkpoint(str(checkpoint_path), device=self.device)
        adapter.eval()

        # Cache layer shapes
        self._layer_shapes[model_name] = adapter.get_layer_shapes()

        self._adapters[model_name] = adapter
        print(f"[GPU] Adapter loaded: {adapter_type}")
        return adapter

    @modal.method()
    def generate_from_mask(
        self,
        model_name: str,
        mask_dict: Dict[str, List],  # Serialized numpy arrays as lists
        class_label: int,
        n_steps: int = 10,
        m_steps: int = 8,
        s_max: float = 80.0,
        s_min: float = 0.002,
        guidance: float = 1.0,
        noise_mode: str = "stochastic",
        extract_layers: Optional[List[str]] = None,
        return_trajectory: bool = True,
    ) -> Optional[List]:
        """Generate with pre-computed mask from CPU.

        Args:
            model_name: Model to use
            mask_dict: {layer_name: [[values]]} — numpy arrays as nested lists
            class_label: Class for generation
            ... generation params ...

        Returns:
            Serialized result tuple or None on error
        """
        import torch
        from diffviews.core.masking import ActivationMasker
        from diffviews.core.generator import generate_with_mask_multistep

        print(f"[GPU] generate_from_mask: model={model_name}, layers={list(mask_dict.keys())}")

        adapter = self._get_or_load_adapter(model_name)
        if adapter is None:
            return None

        # Reconstruct torch tensors from serialized mask
        activation_dict = {}
        for layer_name, arr in mask_dict.items():
            activation_dict[layer_name] = torch.tensor(arr, dtype=torch.float32)

        # Run generation
        masker = ActivationMasker(adapter)
        for layer_name, activation in activation_dict.items():
            masker.set_mask(layer_name, activation)
        masker.register_hooks(list(activation_dict.keys()))

        try:
            result = generate_with_mask_multistep(
                adapter,
                masker,
                class_label=class_label,
                num_steps=n_steps,
                mask_steps=m_steps,
                sigma_max=s_max,
                sigma_min=s_min,
                guidance_scale=guidance,
                noise_mode=noise_mode.replace(" noise", ""),
                num_samples=1,
                device=self.device,
                extract_layers=extract_layers,
                return_trajectory=return_trajectory,
                return_intermediates=True,
                return_noised_inputs=True,
            )
        finally:
            masker.remove_hooks()

        return self._serialize_tuple(result)

    @modal.method()
    def get_layer_shapes(self, model_name: str) -> Optional[Dict[str, Tuple]]:
        """Get layer shapes for a model (for CPU mask computation)."""
        adapter = self._get_or_load_adapter(model_name)
        if adapter is None:
            return None
        return self._layer_shapes.get(model_name)

    @modal.method()
    def health_check(self) -> Dict:
        """Check worker health."""
        import torch
        return {
            "status": "ok",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cached_adapters": list(self._adapters.keys()),
        }

    def _serialize_tuple(self, result: tuple) -> list:
        """Convert torch tensors in tuple to numpy for Modal serialization."""
        import numpy as np

        serialized = []
        for value in result:
            if hasattr(value, "cpu"):
                serialized.append(value.cpu().numpy())
            elif isinstance(value, list) and value and hasattr(value[0], "cpu"):
                serialized.append([v.cpu().numpy() for v in value])
            elif isinstance(value, dict):
                serialized.append(self._serialize_dict(value))
            else:
                serialized.append(value)
        return serialized

    def _serialize_dict(self, d: dict) -> dict:
        """Convert torch tensors in dict to numpy."""
        import numpy as np

        serialized = {}
        for key, value in d.items():
            if hasattr(value, "cpu"):
                serialized[key] = value.cpu().numpy()
            elif isinstance(value, list) and value and hasattr(value[0], "cpu"):
                serialized[key] = [v.cpu().numpy() for v in value]
            else:
                serialized[key] = value
        return serialized


@app.local_entrypoint()
def main():
    """Test GPU worker locally."""
    worker = GPUWorker()
    status = worker.health_check.remote()
    print(f"GPU Worker status: {status}")

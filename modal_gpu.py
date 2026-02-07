"""
Modal GPU worker for diffviews — handles generation + extraction.

Called remotely by modal_web.py (CPU container).
Scales to zero when idle for cost savings.

Usage:
    # Deploy both together
    modal deploy modal_web.py modal_gpu.py
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal

# Shared app name for cross-container calls
app = modal.App("diffviews-gpu")

# GPU image with cuML + torch
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.0",
        "tqdm>=4.60.0",
        "numba==0.58.1",
        "scipy>=1.7.0",
        "plotly>=5.18.0",
        "matplotlib>=3.5.0",
        "boto3>=1.28.0",
        # cuML GPU acceleration
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install("cuml-cu12>=25.02", "cupy-cuda12x>=12.0")
    # TODO: revert to @main before merging
    .pip_install("diffviews @ git+https://github.com/mckellcarter/diffviews.git@feature/modal-migrate")
)

vol = modal.Volume.from_name("diffviews-data", create_if_missing=True)
r2_secret = modal.Secret.from_name("R2_ACCESS")

DATA_DIR = Path("/data")


def _get_device() -> str:
    """Detect GPU device."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/data": vol},
    secrets=[r2_secret],
    timeout=300,
    scaledown_window=60,  # Scale to zero quickly when idle
)
class GPUWorker:
    """Stateful GPU worker — maintains adapter + visualizer between calls."""

    @modal.enter()
    def setup(self):
        """Called once when container starts (cold start)."""
        from diffviews.visualization.visualizer import GradioVisualizer

        print("=" * 50)
        print("DiffViews GPU Worker starting...")
        print("=" * 50)

        self.device = _get_device()
        print(f"  Device: {self.device}")

        # Ensure data is available
        self._ensure_data()

        # Initialize visualizer (lazy adapter loading)
        self.visualizer = GradioVisualizer(data_dir=DATA_DIR, device=self.device)
        print("GPU Worker ready.")

    def _ensure_data(self):
        """Download data if missing."""
        from diffviews.data.r2_cache import R2DataStore

        store = R2DataStore()
        if not store.enabled:
            print("Warning: R2 not configured, assuming data exists")
            return

        for model in ["dmd2", "edm"]:
            config = DATA_DIR / model / "config.json"
            if not config.exists():
                print(f"Downloading {model} data from R2...")
                store.download_model_data(model, DATA_DIR)

        vol.commit()

    @modal.method()
    def generate(
        self,
        model_name: str,
        neighbor_indices: List[int],
        class_label: int,
        n_steps: int = 10,
        m_steps: int = 8,
        s_max: float = 80.0,
        s_min: float = 0.002,
        guidance: float = 1.0,
        noise_mode: str = "stochastic",
        extract_layers: Optional[List[str]] = None,
        can_project: bool = True,
    ) -> Optional[Dict]:
        """Run masked generation on GPU.

        Returns dict with 'images', 'trajectory', 'intermediates', etc.
        """
        from diffviews.core.masking import ActivationMasker
        from diffviews.core.generator import generate_with_mask_multistep

        print(f"[GPU] generate: model={model_name}, neighbors={len(neighbor_indices)}")

        # Ensure model loaded
        if not self.visualizer._ensure_model_loaded(model_name):
            print(f"[GPU] Failed to load model {model_name}")
            return None

        with self.visualizer._generation_lock:
            adapter = self.visualizer.load_adapter(model_name)
            if adapter is None:
                return None

            activation_dict = self.visualizer.prepare_activation_dict(
                model_name, neighbor_indices
            )
            if activation_dict is None:
                return None

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
                    extract_layers=extract_layers if can_project else None,
                    return_trajectory=can_project,
                    return_intermediates=True,
                    return_noised_inputs=True,
                )
            finally:
                masker.remove_hooks()

        # Convert tensors to numpy for serialization (preserve tuple structure)
        return self._serialize_tuple(result)

    def _serialize_tuple(self, result: tuple) -> list:
        """Convert torch tensors in tuple to numpy for Modal serialization."""
        import numpy as np

        serialized = []
        for value in result:
            if hasattr(value, "cpu"):  # torch tensor
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

    @modal.method()
    def extract_layer(
        self,
        model_name: str,
        layer_name: str,
        batch_size: int = 32,
    ) -> Optional[Tuple[List, List]]:
        """Extract layer activations for all samples.

        Returns (activations_list, sample_ids) or None on failure.
        """
        print(f"[GPU] extract_layer: model={model_name}, layer={layer_name}")

        if not self.visualizer._ensure_model_loaded(model_name):
            return None

        result = self.visualizer.extract_layer_activations(
            model_name, layer_name, batch_size
        )
        return result

    @modal.method()
    def compute_umap(
        self,
        activations: List[List[float]],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        pca_components: Optional[int] = 50,
    ) -> Optional[List[List[float]]]:
        """Compute UMAP embeddings using cuML GPU.

        Returns 2D embeddings as list of [x, y] pairs.
        """
        import numpy as np
        from diffviews.processing.umap import compute_umap

        print(f"[GPU] compute_umap: {len(activations)} samples")

        activations_np = np.array(activations, dtype=np.float32)
        embeddings, _, _, _ = compute_umap(
            activations_np,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            pca_components=pca_components,
        )
        return embeddings.tolist()

    @modal.method()
    def health_check(self) -> Dict:
        """Check worker health and return status."""
        import torch
        return {
            "status": "ok",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "models_discovered": list(self.visualizer.model_configs.keys()),
        }


# For direct testing
@app.local_entrypoint()
def main():
    """Test GPU worker locally."""
    worker = GPUWorker()
    status = worker.health_check.remote()
    print(f"GPU Worker status: {status}")

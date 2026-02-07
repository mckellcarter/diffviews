"""
GPU operations for diffviews visualization.

Contains the module-level visualizer reference and GPU wrapper functions
that can be overridden by HF Spaces @spaces.GPU decorator or Modal.

Supports hybrid CPU/GPU mode via remote GPU worker.
"""

from diffviews.core.masking import ActivationMasker
from diffviews.core.generator import generate_with_mask_multistep


# Module-level visualizer reference for GPU functions (set by create_gradio_app).
_app_visualizer = None

# Optional remote GPU worker for hybrid CPU/GPU mode
_remote_gpu_worker = None


def set_visualizer(visualizer):
    """Set the global visualizer reference for GPU operations."""
    global _app_visualizer
    _app_visualizer = visualizer


def get_visualizer():
    """Get the current visualizer, raising if not initialized."""
    if _app_visualizer is None:
        raise RuntimeError("Visualizer not initialized. Call set_visualizer() first.")
    return _app_visualizer


def set_remote_gpu_worker(worker):
    """Set remote GPU worker for hybrid CPU/GPU mode.

    When set, GPU operations are dispatched to the remote worker
    instead of running locally.
    """
    global _remote_gpu_worker
    _remote_gpu_worker = worker
    print(f"[gpu_ops] Remote GPU worker configured: {worker}")


def is_hybrid_mode() -> bool:
    """Check if running in hybrid CPU/GPU mode."""
    return _remote_gpu_worker is not None


def _generate_on_gpu(
    model_name, all_neighbors, class_label,
    n_steps, m_steps, s_max, s_min, guidance, noise_mode,
    extract_layers, can_project
):
    """Run masked generation on GPU.

    In hybrid mode, dispatches to remote GPU worker.
    Otherwise, runs locally with _app_visualizer.
    """
    # Hybrid mode: dispatch to remote GPU worker
    if _remote_gpu_worker is not None:
        print(f"[gpu_ops] Dispatching generation to remote GPU worker...")
        result = _remote_gpu_worker.generate.remote(
            model_name=model_name,
            neighbor_indices=list(all_neighbors),
            class_label=int(class_label),
            n_steps=int(n_steps),
            m_steps=int(m_steps),
            s_max=float(s_max),
            s_min=float(s_min),
            guidance=float(guidance),
            noise_mode=(noise_mode or "stochastic noise").replace(" noise", ""),
            extract_layers=extract_layers if can_project else None,
            can_project=can_project,
        )
        # Convert numpy arrays back to torch tensors if needed
        return _deserialize_result(result)

    # Local mode: run on this machine's GPU
    visualizer = _app_visualizer
    with visualizer._generation_lock:
        adapter = visualizer.load_adapter(model_name)
        if adapter is None:
            return None

        activation_dict = visualizer.prepare_activation_dict(model_name, all_neighbors)
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
                num_steps=int(n_steps),
                mask_steps=int(m_steps),
                sigma_max=float(s_max),
                sigma_min=float(s_min),
                guidance_scale=float(guidance),
                noise_mode=(noise_mode or "stochastic noise").replace(" noise", ""),
                num_samples=1,
                device=visualizer.device,
                extract_layers=extract_layers if can_project else None,
                return_trajectory=can_project,
                return_intermediates=True,
                return_noised_inputs=True,
            )
        finally:
            masker.remove_hooks()

    return result


def _deserialize_result(result):
    """Convert numpy arrays from remote GPU worker back to torch tensors."""
    if result is None:
        return None

    import torch
    import numpy as np

    deserialized = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            deserialized[key] = torch.from_numpy(value)
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            deserialized[key] = [torch.from_numpy(v) for v in value]
        elif isinstance(value, dict):
            deserialized[key] = _deserialize_result(value)
        else:
            deserialized[key] = value
    return deserialized


def _extract_layer_on_gpu(model_name, layer_name, batch_size=32):
    """Extract layer activations on GPU.

    In hybrid mode, dispatches to remote GPU worker.
    Otherwise, runs locally with _app_visualizer.
    """
    # Hybrid mode: dispatch to remote GPU worker
    if _remote_gpu_worker is not None:
        print(f"[gpu_ops] Dispatching layer extraction to remote GPU worker...")
        return _remote_gpu_worker.extract_layer.remote(
            model_name=model_name,
            layer_name=layer_name,
            batch_size=batch_size,
        )

    # Local mode: run on this machine's GPU
    return _app_visualizer.extract_layer_activations(model_name, layer_name, batch_size)

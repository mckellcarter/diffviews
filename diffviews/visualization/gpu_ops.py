"""
GPU operations for diffviews visualization.

Contains the module-level visualizer reference and GPU wrapper functions
that can be overridden by HF Spaces @spaces.GPU decorator or Modal.

Supports hybrid CPU/GPU mode via remote GPU worker.
"""

from diffviews.core.masking import ActivationMasker, compute_mask_dict
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

    In hybrid mode: computes mask on CPU, dispatches to remote GPU worker.
    Otherwise, runs locally with _app_visualizer.
    """
    # Hybrid mode: compute mask on CPU, send to GPU
    if _remote_gpu_worker is not None:
        visualizer = _app_visualizer
        model_data = visualizer.get_model(model_name)
        if model_data is None or model_data.activations is None:
            print(f"[gpu_ops] Model {model_name} not loaded or no activations")
            return None

        # Get layer shapes (needed for mask computation)
        layers = sorted(model_data.umap_params.get("layers", ["encoder_bottleneck", "midblock"]))
        if not model_data.layer_shapes:
            # Fetch from GPU worker if not cached locally
            print(f"[gpu_ops] Fetching layer shapes from GPU...")
            model_data.layer_shapes = _remote_gpu_worker.get_layer_shapes.remote(model_name)

        if not model_data.layer_shapes:
            print(f"[gpu_ops] No layer shapes available")
            return None

        # Compute mask on CPU (lightweight, ~32KB)
        print(f"[gpu_ops] Computing mask on CPU...")
        mask_dict = compute_mask_dict(
            model_data.activations,
            list(all_neighbors),
            model_data.layer_shapes,
            layers,
        )
        if not mask_dict:
            print(f"[gpu_ops] Failed to compute mask")
            return None

        # Serialize mask to nested lists for Modal
        mask_serialized = {k: v.tolist() for k, v in mask_dict.items()}

        print(f"[gpu_ops] Dispatching to GPU worker (mask: {list(mask_dict.keys())})")
        result = _remote_gpu_worker.generate_from_mask.remote(
            model_name=model_name,
            mask_dict=mask_serialized,
            class_label=int(class_label),
            n_steps=int(n_steps),
            m_steps=int(m_steps),
            s_max=float(s_max),
            s_min=float(s_min),
            guidance=float(guidance),
            noise_mode=(noise_mode or "stochastic noise").replace(" noise", ""),
            extract_layers=extract_layers if can_project else None,
            return_trajectory=can_project,
        )
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
    """Convert numpy arrays from remote GPU worker back to torch tensors.

    Handles list (serialized tuple) or dict structures.
    """
    if result is None:
        return None

    import torch
    import numpy as np

    # Handle list (serialized tuple from generate)
    if isinstance(result, list):
        deserialized = []
        for value in result:
            if isinstance(value, np.ndarray):
                deserialized.append(torch.from_numpy(value))
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                deserialized.append([torch.from_numpy(v) for v in value])
            elif isinstance(value, dict):
                deserialized.append(_deserialize_dict(value))
            else:
                deserialized.append(value)
        return tuple(deserialized)

    # Handle dict
    if isinstance(result, dict):
        return _deserialize_dict(result)

    return result


def _deserialize_dict(d):
    """Convert numpy arrays in dict back to torch tensors."""
    import torch
    import numpy as np

    deserialized = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            deserialized[key] = torch.from_numpy(value)
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            deserialized[key] = [torch.from_numpy(v) for v in value]
        else:
            deserialized[key] = value
    return deserialized


def _extract_layer_on_gpu(model_name, layer_name, batch_size=32):
    """Extract layer activations on GPU.

    In hybrid mode: not supported (lightweight GPU worker doesn't have extract).
    Pre-seed all layers to R2 to avoid this.
    Otherwise, runs locally with _app_visualizer.
    """
    # Hybrid mode: lightweight GPU worker doesn't support extraction
    if _remote_gpu_worker is not None:
        print(f"[gpu_ops] Layer extraction not supported in hybrid mode.")
        print(f"[gpu_ops] Pre-seed layer '{layer_name}' to R2 to use it.")
        return None

    # Local mode: run on this machine's GPU
    return _app_visualizer.extract_layer_activations(model_name, layer_name, batch_size)

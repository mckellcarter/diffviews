"""
GPU operations for diffviews visualization.

Contains the module-level visualizer reference and GPU wrapper functions
that can be overridden by HF Spaces @spaces.GPU decorator or Modal.
"""

from diffviews.core.masking import ActivationMasker
from diffviews.core.generator import generate_with_mask_multistep


# Module-level visualizer reference for GPU functions (set by create_gradio_app).
_app_visualizer = None


def set_visualizer(visualizer):
    """Set the global visualizer reference for GPU operations."""
    global _app_visualizer
    _app_visualizer = visualizer


def get_visualizer():
    """Get the current visualizer, raising if not initialized."""
    if _app_visualizer is None:
        raise RuntimeError("Visualizer not initialized. Call set_visualizer() first.")
    return _app_visualizer


def _generate_on_gpu(
    model_name, all_neighbors, class_label,
    n_steps, m_steps, s_max, s_min, guidance, noise_mode,
    extract_layers, can_project
):
    """Run masked generation on GPU."""
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


def _extract_layer_on_gpu(model_name, layer_name, batch_size=32):
    """Extract layer activations on GPU."""
    return _app_visualizer.extract_layer_activations(model_name, layer_name, batch_size)

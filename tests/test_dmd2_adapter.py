"""Tests for DMD2ImageNetAdapter."""

import pytest
import torch

from diffviews.adapters import get_adapter, list_adapters


# Skip all tests if DMD2 adapter not installed
pytestmark = pytest.mark.skipif(
    'dmd2-imagenet-64' not in list_adapters(),
    reason="DMD2 adapter not installed"
)


@pytest.fixture(scope="module")
def checkpoint_path():
    """Path to test checkpoint. Override via --checkpoint flag or env var."""
    import os
    path = os.environ.get(
        'DMD2_CHECKPOINT',
        '/Users/mckell/Documents/dmd2_data/training_data/10step_model_train_out/imagenet_10step_denoising/checkpoint_model_499500'
    )
    if not os.path.exists(path):
        pytest.skip(f"Checkpoint not found: {path}")
    return path


@pytest.fixture(scope="module")
def adapter(checkpoint_path):
    """Load adapter once for all tests."""
    AdapterClass = get_adapter('dmd2-imagenet-64')
    return AdapterClass.from_checkpoint(checkpoint_path, device='cpu')


class TestAdapterDiscovery:
    """Test adapter registration and discovery."""

    def test_adapter_in_registry(self):
        adapters = list_adapters()
        assert 'dmd2-imagenet-64' in adapters

    def test_get_adapter_returns_class(self):
        AdapterClass = get_adapter('dmd2-imagenet-64')
        assert AdapterClass is not None
        assert hasattr(AdapterClass, 'from_checkpoint')


class TestAdapterLoading:
    """Test checkpoint loading and properties."""

    def test_from_checkpoint(self, adapter):
        assert adapter is not None

    def test_model_type(self, adapter):
        assert adapter.model_type == 'dmd2-imagenet-64'

    def test_resolution(self, adapter):
        assert adapter.resolution == 64

    def test_num_classes(self, adapter):
        assert adapter.num_classes == 1000

    def test_hookable_layers(self, adapter):
        layers = adapter.hookable_layers
        assert len(layers) > 0
        assert 'encoder_bottleneck' in layers
        assert 'midblock' in layers

    def test_layer_shapes(self, adapter):
        shapes = adapter.get_layer_shapes()
        assert len(shapes) > 0
        # Bottleneck should be (768, 8, 8) for ImageNet 64
        assert shapes['encoder_bottleneck'] == (768, 8, 8)
        assert shapes['midblock'] == (768, 8, 8)


class TestForwardPass:
    """Test forward pass functionality."""

    def test_forward_shape(self, adapter):
        x = torch.randn(1, 3, 64, 64)
        sigma = torch.ones(1) * 80.0
        labels = torch.zeros(1, 1000)
        labels[0, 0] = 1.0

        with torch.no_grad():
            out = adapter.forward(x * 80.0, sigma, labels)

        assert out.shape == (1, 3, 64, 64)

    def test_forward_output_range(self, adapter):
        x = torch.randn(1, 3, 64, 64)
        sigma = torch.ones(1) * 80.0
        labels = torch.zeros(1, 1000)
        labels[0, 0] = 1.0

        with torch.no_grad():
            out = adapter.forward(x * 80.0, sigma, labels)

        # Output should be roughly in [-1, 1] range for denoised images
        assert out.min() > -2.0
        assert out.max() < 2.0

    def test_forward_batch(self, adapter):
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)
        sigma = torch.ones(batch_size) * 80.0
        labels = torch.zeros(batch_size, 1000)
        for i in range(batch_size):
            labels[i, i * 100] = 1.0

        with torch.no_grad():
            out = adapter.forward(x * 80.0, sigma, labels)

        assert out.shape == (batch_size, 3, 64, 64)


class TestExtractionHooks:
    """Test activation extraction hooks."""

    def test_extraction_single_layer(self, adapter):
        layer = 'encoder_bottleneck'
        adapter.register_activation_hooks([layer], adapter.make_extraction_hook(layer))

        x = torch.randn(1, 3, 64, 64)
        sigma = torch.ones(1) * 80.0
        labels = torch.zeros(1, 1000)
        labels[0, 0] = 1.0

        with torch.no_grad():
            adapter.forward(x * 80.0, sigma, labels)

        act = adapter.get_activation(layer)
        assert act is not None
        assert act.shape == (1, 768, 8, 8)

        adapter.remove_hooks()
        adapter.clear_activations()

    def test_extraction_multiple_layers(self, adapter):
        layers = ['encoder_bottleneck', 'midblock']
        for layer in layers:
            adapter.register_activation_hooks([layer], adapter.make_extraction_hook(layer))

        x = torch.randn(1, 3, 64, 64)
        sigma = torch.ones(1) * 80.0
        labels = torch.zeros(1, 1000)
        labels[0, 0] = 1.0

        with torch.no_grad():
            adapter.forward(x * 80.0, sigma, labels)

        acts = adapter.get_activations()
        assert len(acts) == 2
        for layer in layers:
            assert layer in acts
            assert acts[layer].shape == (1, 768, 8, 8)

        adapter.remove_hooks()
        adapter.clear_activations()

    def test_hook_cleanup(self, adapter):
        layer = 'encoder_bottleneck'
        adapter.register_activation_hooks([layer], adapter.make_extraction_hook(layer))
        assert adapter.num_hooks > 0

        adapter.remove_hooks()
        assert adapter.num_hooks == 0


class TestMaskingHooks:
    """Test activation masking hooks."""

    def test_zero_mask_changes_output(self, adapter):
        layer = 'encoder_bottleneck'
        x = torch.randn(1, 3, 64, 64)
        sigma = torch.ones(1) * 80.0
        labels = torch.zeros(1, 1000)
        labels[0, 0] = 1.0

        # Get original output
        with torch.no_grad():
            out_original = adapter.forward(x * 80.0, sigma, labels)

        # Get layer shape and create zero mask
        shapes = adapter.get_layer_shapes()
        mask = torch.zeros(1, *shapes[layer])

        # Apply mask
        adapter.register_activation_hooks([layer], adapter.make_mask_hook(layer, mask))
        with torch.no_grad():
            out_masked = adapter.forward(x * 80.0, sigma, labels)
        adapter.remove_hooks()

        # Outputs should differ
        diff = (out_original - out_masked).abs().mean().item()
        assert diff > 0.01, "Zero mask should change output"

    def test_random_mask_changes_output(self, adapter):
        layer = 'encoder_bottleneck'
        x = torch.randn(1, 3, 64, 64)
        sigma = torch.ones(1) * 80.0
        labels = torch.zeros(1, 1000)
        labels[0, 0] = 1.0

        # Get original output
        with torch.no_grad():
            out_original = adapter.forward(x * 80.0, sigma, labels)

        # Create random mask
        shapes = adapter.get_layer_shapes()
        mask = torch.randn(1, *shapes[layer]) * 0.5

        # Apply mask
        adapter.register_activation_hooks([layer], adapter.make_mask_hook(layer, mask))
        with torch.no_grad():
            out_masked = adapter.forward(x * 80.0, sigma, labels)
        adapter.remove_hooks()

        # Outputs should differ
        diff = (out_original - out_masked).abs().mean().item()
        assert diff > 0.01, "Random mask should change output"

    def test_cross_sample_masking(self, adapter):
        """Masking sample B with A's activation should make B closer to A."""
        layer = 'encoder_bottleneck'
        sigma = torch.ones(1) * 80.0

        # Sample A
        x_a = torch.randn(1, 3, 64, 64)
        labels_a = torch.zeros(1, 1000)
        labels_a[0, 207] = 1.0  # golden retriever

        # Sample B
        x_b = torch.randn(1, 3, 64, 64)
        labels_b = torch.zeros(1, 1000)
        labels_b[0, 281] = 1.0  # tabby cat

        # Extract A's activation
        adapter.register_activation_hooks([layer], adapter.make_extraction_hook(layer))
        with torch.no_grad():
            out_a = adapter.forward(x_a * 80.0, sigma, labels_a)
        act_a = adapter.get_activation(layer).clone()
        adapter.remove_hooks()
        adapter.clear_activations()

        # Generate B normally
        with torch.no_grad():
            out_b_normal = adapter.forward(x_b * 80.0, sigma, labels_b)

        # Generate B with A's activation
        adapter.register_activation_hooks([layer], adapter.make_mask_hook(layer, act_a))
        with torch.no_grad():
            out_b_masked = adapter.forward(x_b * 80.0, sigma, labels_b)
        adapter.remove_hooks()

        # Masked B should be closer to A than normal B
        diff_normal = (out_a - out_b_normal).abs().mean().item()
        diff_masked = (out_a - out_b_masked).abs().mean().item()
        assert diff_masked < diff_normal, "Masked output should be closer to source"

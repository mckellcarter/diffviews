"""
Unit tests for diffviews.core.masking
Adapted from DMD2/visualizer/test_activation_masking.py
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from diffviews.core.masking import (
    ActivationMasker,
    load_activation_from_npz,
    unflatten_activation
)
from diffviews.adapters.base import GeneratorAdapter


class MockAdapter(GeneratorAdapter):
    """Mock adapter for testing masking without real model."""

    def __init__(self):
        self._hooks = []
        self._modules = {
            'encoder_bottleneck': MockModule(),
            'midblock': MockModule(),
            'decoder_block_0': MockModule(),
        }

    @property
    def model_type(self) -> str:
        return 'mock'

    @property
    def resolution(self) -> int:
        return 64

    @property
    def num_classes(self) -> int:
        return 1000

    @property
    def hookable_layers(self):
        return list(self._modules.keys())

    def forward(self, x, sigma, class_labels=None, **kwargs):
        return torch.randn_like(x)

    def register_activation_hooks(self, layer_names, hook_fn):
        handles = []
        for name in layer_names:
            if name in self._modules:
                handle = self._modules[name].register_forward_hook(hook_fn)
                handles.append(handle)
        return handles

    def get_layer_shapes(self):
        return {
            'encoder_bottleneck': (256, 8, 8),
            'midblock': (512, 4, 4),
            'decoder_block_0': (256, 8, 8),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda', **kwargs):
        return cls()

    @classmethod
    def get_default_config(cls):
        return {}


class MockModule:
    """Mock PyTorch module for testing hooks."""
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook_fn):
        hook = MockHook(hook_fn)
        self.hooks.append(hook)
        return hook


class MockHook:
    """Mock hook handle."""
    def __init__(self, fn):
        self.fn = fn
        self.removed = False

    def remove(self):
        self.removed = True


class TestActivationMasker:
    """Test ActivationMasker class."""

    def test_initialization(self):
        """Test masker initialization."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        assert masker.adapter is adapter
        assert not masker.masks
        assert not masker._handles

    def test_set_mask(self):
        """Test setting activation masks."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        activation = torch.randn(1, 256, 8, 8)

        masker.set_mask("midblock", activation)

        assert "midblock" in masker.masks
        assert masker.masks["midblock"].shape == activation.shape
        # Should be stored on CPU
        assert masker.masks["midblock"].device.type == "cpu"

    def test_clear_masks(self):
        """Test clearing masks."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("layer1", torch.randn(1, 128, 16, 16))
        masker.set_mask("layer2", torch.randn(1, 256, 8, 8))

        assert len(masker.masks) == 2

        masker.clear_masks()

        assert len(masker.masks) == 0

    def test_masking_hook_replaces_output(self):
        """Test that masking hook replaces layer output."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        fixed_activation = torch.ones(1, 64, 16, 16)
        masker.set_mask("test_layer", fixed_activation)

        # Create hook function via internal method
        hook_fn = masker._make_hook("test_layer")

        # Simulate forward pass
        original_output = torch.zeros(1, 64, 16, 16)
        result = hook_fn(None, None, original_output)

        # Result should be the fixed activation, not zeros
        assert torch.allclose(result, fixed_activation)
        assert not torch.allclose(result, original_output)

    def test_masking_hook_batch_expansion(self):
        """Test that mask expands to batch size."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        fixed_activation = torch.ones(1, 64, 16, 16)
        masker.set_mask("test_layer", fixed_activation)

        hook_fn = masker._make_hook("test_layer")

        # Simulate batch of 4
        batch_output = torch.zeros(4, 64, 16, 16)
        result = hook_fn(None, None, batch_output)

        assert result.shape == (4, 64, 16, 16)
        # All batch items should be the same
        for i in range(4):
            assert torch.allclose(result[i], fixed_activation[0])

    def test_masking_hook_tuple_output(self):
        """Test handling of tuple outputs."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        fixed_activation = torch.ones(1, 64, 16, 16)
        masker.set_mask("test_layer", fixed_activation)

        hook_fn = masker._make_hook("test_layer")

        # Some modules return tuples
        tuple_output = (torch.zeros(1, 64, 16, 16), torch.randn(1, 128))
        result = hook_fn(None, None, tuple_output)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.allclose(result[0], fixed_activation)
        # Second element should be unchanged
        assert torch.equal(result[1], tuple_output[1])

    def test_masking_hook_no_mask_passthrough(self):
        """Test that hook passes through when no mask set."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        hook_fn = masker._make_hook("nonexistent_layer")

        original_output = torch.randn(1, 64, 16, 16)
        result = hook_fn(None, None, original_output)

        # Should pass through unchanged
        assert torch.equal(result, original_output)

    def test_register_hooks(self):
        """Test hook registration via adapter."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("encoder_bottleneck", torch.randn(1, 256, 8, 8))
        masker.set_mask("midblock", torch.randn(1, 512, 4, 4))

        masker.register_hooks(["encoder_bottleneck", "midblock"])

        assert len(masker._handles) == 2

    def test_remove_hooks(self):
        """Test hook removal."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("encoder_bottleneck", torch.randn(1, 256, 8, 8))

        masker.register_hooks(["encoder_bottleneck"])
        assert len(masker._handles) == 1

        masker.remove_hooks()
        assert len(masker._handles) == 0

    def test_context_manager(self):
        """Test context manager support."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask("encoder_bottleneck", torch.randn(1, 256, 8, 8))

        with masker as m:
            assert len(m._handles) > 0

        # Hooks should be removed after context exit
        assert len(masker._handles) == 0


class TestActivationLoading:
    """Test activation loading utilities."""

    def test_load_activation_from_npz(self):
        """Test loading activation from NPZ file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test_activation.npz"

            # Create test activation
            test_activation = np.random.randn(1, 256, 8, 8).astype(np.float32)
            np.savez(npz_path, midblock=test_activation)

            # Load activation
            loaded = load_activation_from_npz(npz_path, "midblock")

            assert isinstance(loaded, torch.Tensor)
            assert loaded.shape == (1, 256, 8, 8)
            assert torch.allclose(loaded, torch.from_numpy(test_activation))

    def test_load_activation_missing_layer(self):
        """Test error handling for missing layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test_activation.npz"

            # Create NPZ with different layer
            test_activation = np.random.randn(1, 128, 16, 16).astype(np.float32)
            np.savez(npz_path, encoder=test_activation)

            # Try to load nonexistent layer
            with pytest.raises(ValueError, match="not found"):
                load_activation_from_npz(npz_path, "midblock")

    def test_load_activation_adds_batch_dim(self):
        """Test that batch dimension is added if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test_activation.npz"

            # Create 1D activation (missing batch dim)
            test_activation = np.random.randn(2048).astype(np.float32)
            np.savez(npz_path, flat_layer=test_activation)

            loaded = load_activation_from_npz(npz_path, "flat_layer")

            assert loaded.shape == (1, 2048)


class TestUnflattenActivation:
    """Test activation unflattening."""

    def test_unflatten_basic(self):
        """Test basic unflattening."""
        flat = torch.randn(1, 256 * 8 * 8)
        target_shape = (256, 8, 8)

        unflattened = unflatten_activation(flat, target_shape)

        assert unflattened.shape == (1, 256, 8, 8)

    def test_unflatten_1d_input(self):
        """Test unflattening with 1D input."""
        flat = torch.randn(128 * 16 * 16)
        target_shape = (128, 16, 16)

        unflattened = unflatten_activation(flat, target_shape)

        assert unflattened.shape == (1, 128, 16, 16)

    def test_unflatten_preserves_values(self):
        """Test that values are preserved during reshape."""
        C, H, W = 64, 4, 4

        original = torch.arange(C * H * W).float().reshape(1, C, H, W)
        flat = original.reshape(1, -1)

        unflattened = unflatten_activation(flat, (C, H, W))

        assert torch.equal(unflattened, original)

    def test_unflatten_different_sizes(self):
        """Test various spatial sizes."""
        test_cases = [
            ((512, 4, 4), 512 * 4 * 4),
            ((256, 8, 8), 256 * 8 * 8),
            ((128, 16, 16), 128 * 16 * 16),
            ((64, 32, 32), 64 * 32 * 32),
        ]

        for target_shape, flat_size in test_cases:
            flat = torch.randn(1, flat_size)
            unflattened = unflatten_activation(flat, target_shape)

            assert unflattened.shape == (1,) + target_shape


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_save_load_mask_cycle(self):
        """Test saving, loading, and applying mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test_activation.npz"

            # Create and save activation
            original_activation = torch.randn(1, 256, 8, 8)
            np.savez(
                npz_path,
                midblock=original_activation.numpy()
            )

            # Load activation
            loaded = load_activation_from_npz(npz_path, "midblock")

            # Create masker
            adapter = MockAdapter()
            masker = ActivationMasker(adapter)
            masker.set_mask("midblock", loaded)

            # Verify mask is set correctly
            assert "midblock" in masker.masks
            assert torch.allclose(masker.masks["midblock"], original_activation)

    def test_flatten_unflatten_cycle(self):
        """Test flattening and unflattening preserves data."""
        original = torch.randn(1, 256, 8, 8)
        flat = original.reshape(1, -1)
        restored = unflatten_activation(flat, (256, 8, 8))

        assert torch.allclose(original, restored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

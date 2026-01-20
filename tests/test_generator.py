"""
Unit tests for diffviews.core.generator
Adapted from DMD2/visualizer/test_generate_from_activation.py
"""

import pytest
import torch
from pathlib import Path
from PIL import Image
import tempfile
from typing import Dict, List, Tuple, Any

from diffviews.core.generator import (
    generate_with_mask,
    generate_with_mask_multistep,
    save_generated_sample,
    get_denoising_sigmas,
    tensor_to_uint8_image
)
from diffviews.core.masking import ActivationMasker
from diffviews.adapters.base import GeneratorAdapter


class MockAdapter(GeneratorAdapter):
    """Mock adapter for testing generation without real model."""

    def __init__(self, output_shape=(1, 3, 64, 64)):
        self.output_shape = output_shape
        self.calls = []
        self._hooks = []

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
    def hookable_layers(self) -> List[str]:
        return ['encoder_bottleneck', 'midblock']

    def forward(self, x, sigma, class_labels=None, **kwargs):
        self.calls.append({
            'x': x,
            'sigma': sigma,
            'class_labels': class_labels
        })
        batch_size = x.shape[0]
        return torch.randn(batch_size, 3, 64, 64, device=x.device)

    def register_activation_hooks(self, layer_names, hook_fn):
        return []

    def get_layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            'encoder_bottleneck': (256, 8, 8),
            'midblock': (512, 4, 4),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda', **kwargs):
        return cls()

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {}


class TestTensorToUint8Image:
    """Test image tensor conversion."""

    def test_basic_conversion(self):
        """Test basic tensor to uint8 conversion."""
        tensor = torch.zeros(1, 3, 64, 64)  # All zeros -> should become 127/128
        result = tensor_to_uint8_image(tensor)

        assert result.dtype == torch.uint8
        assert result.shape == (1, 64, 64, 3)  # BHWC format
        # (0 + 1) * 127.5 = 127.5, clamped to 127 or 128
        assert result.min() >= 0
        assert result.max() <= 255

    def test_full_range(self):
        """Test conversion with full [-1, 1] range."""
        tensor = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]])  # (1, 3, 1, 1)
        result = tensor_to_uint8_image(tensor)

        assert result.shape == (1, 1, 1, 3)
        # -1 -> 0, 0 -> 127/128, 1 -> 255
        assert result[0, 0, 0, 0] == 0      # -1 -> 0
        assert result[0, 0, 0, 2] == 255    # 1 -> 255


class TestDenoisingSigmas:
    """Test sigma schedule generation."""

    def test_basic_schedule(self):
        """Test basic sigma schedule."""
        sigmas = get_denoising_sigmas(4, sigma_max=80.0, sigma_min=0.002)

        assert len(sigmas) == 4
        assert sigmas[0] > sigmas[-1]  # Descending
        assert sigmas[0].item() == pytest.approx(80.0, rel=0.01)

    def test_schedule_length(self):
        """Test different schedule lengths."""
        for num_steps in [1, 4, 10, 20]:
            sigmas = get_denoising_sigmas(num_steps, 80.0, 0.002)
            assert len(sigmas) == num_steps


class TestGenerateWithMask:
    """Test single-step generation with masked activations."""

    def test_basic_generation(self):
        """Test basic generation call."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask(
            adapter,
            masker,
            class_label=42,
            num_samples=1,
            device='cpu'
        )

        # Check output shapes
        assert images.shape == (1, 64, 64, 3)
        assert images.dtype == torch.uint8
        assert labels.shape == (1,)
        assert labels[0] == 42

        # Check adapter was called
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call['x'].shape == (1, 3, 64, 64)
        assert call['class_labels'].shape == (1, 1000)
        assert call['class_labels'][0, 42] == 1.0

    def test_batch_generation(self):
        """Test generating multiple samples."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask(
            adapter,
            masker,
            class_label=100,
            num_samples=4,
            device='cpu'
        )

        assert images.shape == (4, 64, 64, 3)
        assert labels.shape == (4,)
        assert torch.all(labels == 100)

    def test_random_labels(self):
        """Test generation with random class labels."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask(
            adapter,
            masker,
            class_label=None,  # Random labels
            num_samples=3,
            device='cpu'
        )

        assert images.shape == (3, 64, 64, 3)
        assert labels.shape == (3,)
        # Labels should be in valid range
        assert torch.all(labels >= 0)
        assert torch.all(labels < 1000)

    def test_uniform_labels(self):
        """Test generation with uniform class distribution."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask(
            adapter,
            masker,
            class_label=-1,  # Uniform
            num_samples=2,
            device='cpu'
        )

        assert images.shape == (2, 64, 64, 3)
        # Labels should be -1 for uniform
        assert torch.all(labels == -1)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        adapter1 = MockAdapter()
        masker1 = ActivationMasker(adapter1)

        images1, labels1 = generate_with_mask(
            adapter1,
            masker1,
            class_label=None,
            seed=42,
            num_samples=1,
            device='cpu'
        )

        adapter2 = MockAdapter()
        masker2 = ActivationMasker(adapter2)

        images2, labels2 = generate_with_mask(
            adapter2,
            masker2,
            class_label=None,
            seed=42,
            num_samples=1,
            device='cpu'
        )

        # Labels should be same with same seed
        assert labels1[0] == labels2[0]

    def test_image_value_range(self):
        """Test that output images are in valid uint8 range."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, _ = generate_with_mask(
            adapter,
            masker,
            num_samples=1,
            device='cpu'
        )

        assert images.dtype == torch.uint8
        assert images.min() >= 0
        assert images.max() <= 255


class TestGenerateWithMaskMultistep:
    """Test multi-step generation."""

    def test_basic_multistep(self):
        """Test basic multi-step generation."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask_multistep(
            adapter,
            masker,
            class_label=42,
            num_steps=4,
            num_samples=1,
            device='cpu'
        )

        assert images.shape == (1, 64, 64, 3)
        assert images.dtype == torch.uint8
        assert labels[0] == 42
        # Should have 4 calls (one per step)
        assert len(adapter.calls) == 4

    def test_multistep_with_trajectory(self):
        """Test multi-step with trajectory extraction."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        result = generate_with_mask_multistep(
            adapter,
            masker,
            class_label=42,
            num_steps=4,
            num_samples=1,
            device='cpu',
            return_trajectory=True,
            extract_layers=['encoder_bottleneck']
        )

        images, labels, trajectory = result  # pylint: disable=unbalanced-tuple-unpacking
        assert images.shape == (1, 64, 64, 3)
        # Trajectory may be empty if no hooks actually extract
        assert isinstance(trajectory, list)

    def test_multistep_with_intermediates(self):
        """Test multi-step with intermediate images."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        result = generate_with_mask_multistep(
            adapter,
            masker,
            class_label=42,
            num_steps=4,
            num_samples=1,
            device='cpu',
            return_intermediates=True
        )

        images, labels, intermediates = result  # pylint: disable=unbalanced-tuple-unpacking
        assert len(intermediates) == 4  # One per step

    def test_mask_steps_parameter(self):
        """Test mask_steps limits masking."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)
        masker.set_mask('midblock', torch.randn(1, 512, 4, 4))

        images, labels = generate_with_mask_multistep(
            adapter,
            masker,
            class_label=42,
            num_steps=4,
            mask_steps=1,  # Only mask first step
            num_samples=1,
            device='cpu'
        )

        assert images.shape == (1, 64, 64, 3)


class TestSaveGeneratedSample:
    """Test saving generated samples."""

    def test_save_basic(self):
        """Test basic sample saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
            activations = {
                'midblock': torch.randn(1, 256, 8, 8)
            }
            metadata = {
                'sample_id': 'test_sample',
                'class_label': 42,
                'model': 'imagenet'
            }

            record = save_generated_sample(
                image,
                activations,
                metadata,
                output_dir,
                'test_sample'
            )

            assert record['sample_id'] == 'test_sample'
            assert record['class_label'] == 42
            assert 'image_path' in record

            # Check image was saved
            image_path = output_dir / record['image_path']
            assert image_path.exists()

            loaded_img = Image.open(image_path)
            assert loaded_img.size == (64, 64)

    def test_save_creates_directories(self):
        """Test that necessary directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
            activations = {'layer': torch.randn(1, 128, 16, 16)}
            metadata = {'sample_id': 'test', 'model': 'imagenet'}

            save_generated_sample(
                image,
                activations,
                metadata,
                output_dir,
                'test'
            )

            assert (output_dir / "images" / "imagenet").exists()

    def test_save_empty_activations(self):
        """Test saving with no activations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
            metadata = {'sample_id': 'test', 'model': 'generated'}

            record = save_generated_sample(
                image,
                {},  # Empty activations
                metadata,
                output_dir,
                'test'
            )

            assert record['sample_id'] == 'test'
            image_path = output_dir / record['image_path']
            assert image_path.exists()

    def test_save_multiple_samples(self):
        """Test saving multiple samples to same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            for i in range(3):
                image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
                activations = {'layer': torch.randn(1, 128, 16, 16)}
                metadata = {
                    'sample_id': f'sample_{i:03d}',
                    'model': 'imagenet'
                }

                save_generated_sample(
                    image,
                    activations,
                    metadata,
                    output_dir,
                    f'sample_{i:03d}'
                )

            image_dir = output_dir / "images" / "imagenet"
            assert len(list(image_dir.glob("*.png"))) == 3


class TestIntegration:
    """Integration tests."""

    def test_generate_and_save_workflow(self):
        """Test complete generate and save workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            adapter = MockAdapter()
            masker = ActivationMasker(adapter)
            masker.set_mask("midblock", torch.randn(1, 512, 4, 4))

            # Generate image
            images, labels = generate_with_mask(
                adapter,
                masker,
                class_label=42,
                num_samples=1,
                device='cpu'
            )

            # Save sample
            metadata = {
                'sample_id': 'generated_001',
                'class_label': int(labels[0]),
                'model': 'imagenet',
                'generated_from_neighbors': [1, 2, 3]
            }

            record = save_generated_sample(
                images[0],
                {},
                metadata,
                output_dir,
                'generated_001'
            )

            assert record['sample_id'] == 'generated_001'
            assert record['class_label'] == 42

            image_path = output_dir / record['image_path']
            assert image_path.exists()

            loaded_img = Image.open(image_path)
            assert loaded_img.size == (64, 64)


class TestGradioVisualizerGeneration:
    """Test GradioVisualizer generation methods."""

    def test_load_adapter_no_checkpoint(self):
        """Test load_adapter returns None when checkpoint missing."""
        import tempfile
        import json
        from diffviews.visualization.app import GradioVisualizer, ModelData
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "dmd2"
            model_dir.mkdir()
            (model_dir / "embeddings").mkdir()
            with open(model_dir / "config.json", "w") as f:
                json.dump({"adapter": "dmd2-imagenet-64"}, f)
            # Create minimal embeddings
            import pandas as pd
            df = pd.DataFrame({"umap_x": [0], "umap_y": [0], "sample_id": ["s0"]})
            df.to_csv(model_dir / "embeddings" / "test.csv", index=False)

            # Mock _load_model_data to return ModelData with no checkpoint
            mock_model_data = ModelData(
                name="dmd2",
                data_dir=model_dir,
                adapter_name="dmd2-imagenet-64",
                checkpoint_path=None,  # No checkpoint
                sigma_max=80.0,
                sigma_min=0.5,
                default_steps=5,
                df=df,
            )
            with patch.object(GradioVisualizer, '_load_model_data', return_value=mock_model_data):
                viz = GradioVisualizer(data_dir=root)

            result = viz.load_adapter("dmd2")
            assert result is None

    def test_load_adapter_missing_file(self):
        """Test load_adapter returns None when checkpoint file doesn't exist."""
        import tempfile
        import json
        from diffviews.visualization.app import GradioVisualizer, ModelData
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "dmd2"
            model_dir.mkdir()
            (model_dir / "embeddings").mkdir()
            with open(model_dir / "config.json", "w") as f:
                json.dump({"adapter": "dmd2-imagenet-64"}, f)
            import pandas as pd
            df = pd.DataFrame({"umap_x": [0], "umap_y": [0], "sample_id": ["s0"]})
            df.to_csv(model_dir / "embeddings" / "test.csv", index=False)

            # Mock _load_model_data with nonexistent checkpoint path
            mock_model_data = ModelData(
                name="dmd2",
                data_dir=model_dir,
                adapter_name="dmd2-imagenet-64",
                checkpoint_path=Path("/nonexistent/path/checkpoint.pt"),
                sigma_max=80.0,
                sigma_min=0.5,
                default_steps=5,
                df=df,
            )
            with patch.object(GradioVisualizer, '_load_model_data', return_value=mock_model_data):
                viz = GradioVisualizer(data_dir=root)

            result = viz.load_adapter("dmd2")
            assert result is None

    def test_prepare_activation_dict_no_activations(self):
        """Test prepare_activation_dict returns None when activations missing."""
        import tempfile
        import json
        from diffviews.visualization.app import GradioVisualizer, ModelData
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "dmd2"
            model_dir.mkdir()
            (model_dir / "embeddings").mkdir()
            with open(model_dir / "config.json", "w") as f:
                json.dump({"adapter": "dmd2-imagenet-64"}, f)
            import pandas as pd
            df = pd.DataFrame({"umap_x": [0], "umap_y": [0], "sample_id": ["s0"]})
            df.to_csv(model_dir / "embeddings" / "test.csv", index=False)

            # Mock _load_model_data with no activations
            mock_model_data = ModelData(
                name="dmd2",
                data_dir=model_dir,
                adapter_name="dmd2-imagenet-64",
                checkpoint_path=None,
                sigma_max=80.0,
                sigma_min=0.5,
                default_steps=5,
                df=df,
                activations=None,  # No activations
            )
            with patch.object(GradioVisualizer, '_load_model_data', return_value=mock_model_data):
                viz = GradioVisualizer(data_dir=root)

            result = viz.prepare_activation_dict("dmd2", [0, 1, 2])
            assert result is None

    def test_prepare_activation_dict_empty_neighbors(self):
        """Test prepare_activation_dict returns None with empty neighbor list."""
        import tempfile
        import json
        import numpy as np
        from diffviews.visualization.app import GradioVisualizer, ModelData
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "dmd2"
            model_dir.mkdir()
            (model_dir / "embeddings").mkdir()
            with open(model_dir / "config.json", "w") as f:
                json.dump({"adapter": "dmd2-imagenet-64"}, f)
            import pandas as pd
            df = pd.DataFrame({"umap_x": [0], "umap_y": [0], "sample_id": ["s0"]})
            df.to_csv(model_dir / "embeddings" / "test.csv", index=False)

            # Mock _load_model_data with activations
            mock_model_data = ModelData(
                name="dmd2",
                data_dir=model_dir,
                adapter_name="dmd2-imagenet-64",
                checkpoint_path=None,
                sigma_max=80.0,
                sigma_min=0.5,
                default_steps=5,
                df=df,
                activations=np.random.randn(100, 512).astype(np.float32),
            )
            with patch.object(GradioVisualizer, '_load_model_data', return_value=mock_model_data):
                viz = GradioVisualizer(data_dir=root)

            result = viz.prepare_activation_dict("dmd2", [])  # Empty neighbors
            assert result is None

    def test_prepare_activation_dict_splits_correctly(self):
        """Test prepare_activation_dict splits activations by layer."""
        import tempfile
        import json
        import numpy as np
        from diffviews.visualization.app import GradioVisualizer, ModelData
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "dmd2"
            model_dir.mkdir()
            (model_dir / "embeddings").mkdir()
            with open(model_dir / "config.json", "w") as f:
                json.dump({"adapter": "dmd2-imagenet-64"}, f)
            # Add umap params with layers
            with open(model_dir / "embeddings" / "test.json", "w") as f:
                json.dump({"layers": ["encoder_bottleneck", "midblock"]}, f)
            import pandas as pd
            df = pd.DataFrame({"umap_x": range(10), "umap_y": range(10), "sample_id": [f"s{i}" for i in range(10)]})
            df.to_csv(model_dir / "embeddings" / "test.csv", index=False)

            # layer shapes: encoder_bottleneck (512,4,4)=8192, midblock (512,4,4)=8192
            layer_shapes = {"encoder_bottleneck": (512, 4, 4), "midblock": (512, 4, 4)}
            total_dim = 512 * 4 * 4 * 2

            # Mock _load_model_data with activations and layer shapes
            mock_model_data = ModelData(
                name="dmd2",
                data_dir=model_dir,
                adapter_name="dmd2-imagenet-64",
                checkpoint_path=None,
                sigma_max=80.0,
                sigma_min=0.5,
                default_steps=5,
                df=df,
                activations=np.random.randn(10, total_dim).astype(np.float32),
                umap_params={"layers": ["encoder_bottleneck", "midblock"]},
                layer_shapes=layer_shapes,
            )
            with patch.object(GradioVisualizer, '_load_model_data', return_value=mock_model_data):
                viz = GradioVisualizer(data_dir=root)

            result = viz.prepare_activation_dict("dmd2", [0, 1, 2])

            assert result is not None
            assert "encoder_bottleneck" in result
            assert "midblock" in result
            assert result["encoder_bottleneck"].shape == (1, 512, 4, 4)
            assert result["midblock"].shape == (1, 512, 4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for diffviews.visualization.gradio_app
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from diffviews.visualization.gradio_app import GradioVisualizer, create_gradio_app


def create_model_dir(root: Path, model_name: str, adapter_name: str, num_samples: int = 10):
    """Helper to create minimal model directory structure with sample data."""
    model_dir = root / model_name
    model_dir.mkdir()
    (model_dir / "activations" / "imagenet_real").mkdir(parents=True)
    (model_dir / "metadata" / "imagenet_real").mkdir(parents=True)
    (model_dir / "embeddings").mkdir(parents=True)
    (model_dir / "images" / "imagenet_real").mkdir(parents=True)

    # Create config.json
    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "adapter": adapter_name,
            "sigma_max": 80.0,
            "sigma_min": 0.5,
            "default_steps": 5
        }, f)

    # Create embeddings CSV with sample data
    df = pd.DataFrame({
        "sample_id": [f"sample_{i:06d}" for i in range(num_samples)],
        "umap_x": np.random.randn(num_samples),
        "umap_y": np.random.randn(num_samples),
        "class_label": np.random.randint(0, 10, num_samples),
        "image_path": [f"images/imagenet_real/sample_{i:06d}.png" for i in range(num_samples)],
    })
    df.to_csv(model_dir / "embeddings" / "demo_embeddings.csv", index=False)

    # Create dummy images
    from PIL import Image
    for i in range(num_samples):
        img = Image.new('RGB', (64, 64), color=(i * 25 % 256, 100, 150))
        img.save(model_dir / "images" / "imagenet_real" / f"sample_{i:06d}.png")

    return model_dir


class TestGradioVisualizerInit:
    """Test GradioVisualizer initialization."""

    def test_model_discovery(self):
        """Test that models are discovered correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert 'dmd2' in viz.model_configs
            assert 'edm' in viz.model_configs
            assert viz.current_model == 'dmd2'

    def test_default_model_selection(self):
        """Test that dmd2 is selected as default when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "edm", "edm-imagenet-64")
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz.current_model == 'dmd2'

    def test_initial_model_override(self):
        """Test that initial_model parameter works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root, initial_model="edm")

            assert viz.current_model == 'edm'

    def test_data_loading(self):
        """Test that embeddings are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert len(viz.df) == 20
            assert 'umap_x' in viz.df.columns
            assert 'umap_y' in viz.df.columns
            assert 'class_label' in viz.df.columns


class TestGradioVisualizerSwitchModel:
    """Test model switching functionality."""

    def test_switch_model_success(self):
        """Test successful model switch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)
            create_model_dir(root, "edm", "edm-imagenet-64", num_samples=15)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz.current_model == 'dmd2'
            assert len(viz.df) == 10

            result = viz.switch_model('edm')

            assert result is True
            assert viz.current_model == 'edm'
            assert len(viz.df) == 15

    def test_switch_model_unknown_fails(self):
        """Test that switching to unknown model fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            result = viz.switch_model('unknown')

            assert result is False
            assert viz.current_model == 'dmd2'

    def test_switch_model_resets_state(self):
        """Test that model switch resets adapter state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Set some state
            viz.adapter = MagicMock()
            viz.layer_shapes = {'test': (1, 2, 3)}

            viz.switch_model('edm')

            assert viz.adapter is None
            assert viz.layer_shapes == {}


class TestGetPlotDataframe:
    """Test plot dataframe generation."""

    def test_basic_dataframe(self):
        """Test basic dataframe generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            plot_df = viz.get_plot_dataframe()

            assert len(plot_df) == 10
            assert 'umap_x' in plot_df.columns
            assert 'umap_y' in plot_df.columns
            assert 'highlight' in plot_df.columns
            assert all(plot_df['highlight'] == 'normal')

    def test_selected_highlight(self):
        """Test that selected point is highlighted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            plot_df = viz.get_plot_dataframe(selected_idx=5)

            assert plot_df.loc[5, 'highlight'] == 'selected'
            assert sum(plot_df['highlight'] == 'selected') == 1

    def test_neighbor_highlights(self):
        """Test that neighbors are highlighted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            plot_df = viz.get_plot_dataframe(
                selected_idx=0,
                manual_neighbors=[1, 2],
                knn_neighbors=[3, 4]
            )

            assert plot_df.loc[0, 'highlight'] == 'selected'
            assert plot_df.loc[1, 'highlight'] == 'manual_neighbor'
            assert plot_df.loc[2, 'highlight'] == 'manual_neighbor'
            assert plot_df.loc[3, 'highlight'] == 'knn_neighbor'
            assert plot_df.loc[4, 'highlight'] == 'knn_neighbor'

    def test_class_highlight(self):
        """Test that class filtering highlights correct samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Get a class that exists in the data
            target_class = viz.df['class_label'].iloc[0]
            expected_count = (viz.df['class_label'] == target_class).sum()

            plot_df = viz.get_plot_dataframe(highlighted_class=target_class)

            highlighted_count = (plot_df['highlight'] == 'class_highlight').sum()
            assert highlighted_count == expected_count


class TestClassOptions:
    """Test class options generation."""

    def test_get_class_options(self):
        """Test class options for dropdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            options = viz.get_class_options()

            assert len(options) > 0
            # Options should be (label, value) tuples
            assert all(isinstance(opt, tuple) and len(opt) == 2 for opt in options)

    def test_class_options_sorted(self):
        """Test that class options are sorted by class ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=50)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            options = viz.get_class_options()
            class_ids = [opt[1] for opt in options]

            assert class_ids == sorted(class_ids)


class TestGetImage:
    """Test image loading."""

    def test_get_image_success(self):
        """Test successful image loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            img = viz.get_image("images/imagenet_real/sample_000000.png")

            assert img is not None
            assert isinstance(img, np.ndarray)
            assert img.shape == (64, 64, 3)

    def test_get_image_not_found(self):
        """Test that missing image returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            img = viz.get_image("images/nonexistent.png")

            assert img is None


class TestCreateGradioApp:
    """Test Gradio app creation."""

    def test_app_creation(self):
        """Test that Gradio app is created successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            app = create_gradio_app(viz)

            assert app is not None
            # Gradio Blocks object
            assert hasattr(app, 'launch')
            assert hasattr(app, 'queue')

    def test_app_with_multiple_models(self):
        """Test app creation with multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            app = create_gradio_app(viz)

            assert app is not None
            assert len(viz.model_configs) == 2


class TestNearestNeighbors:
    """Test KNN fitting."""

    def test_fit_nearest_neighbors(self):
        """Test that KNN model is fitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=30)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz.nn_model is None

            viz.fit_nearest_neighbors()

            assert viz.nn_model is not None

    def test_fit_nearest_neighbors_empty_df(self):
        """Test that KNN fitting handles empty dataframe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            viz.df = pd.DataFrame()
            viz.fit_nearest_neighbors()

            # Should not crash, nn_model stays None
            assert viz.nn_model is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

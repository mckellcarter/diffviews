"""
Unit tests for diffviews.visualization.app
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from diffviews.visualization.app import GradioVisualizer, create_gradio_app


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

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert 'dmd2' in viz.model_configs
            assert 'edm' in viz.model_configs
            assert viz.default_model == 'dmd2'

    def test_default_model_selection(self):
        """Test that dmd2 is selected as default when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "edm", "edm-imagenet-64")
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz.default_model == 'dmd2'

    def test_initial_model_override(self):
        """Test that initial_model parameter works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root, initial_model="edm")

            assert viz.default_model == 'edm'

    def test_data_loading(self):
        """Test that embeddings are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            model_data = viz.get_model(viz.default_model)
            assert len(model_data.df) == 20
            assert 'umap_x' in model_data.df.columns
            assert 'umap_y' in model_data.df.columns
            assert 'class_label' in model_data.df.columns


class TestModelDataAccess:
    """Test model data access patterns for thread safety."""

    def test_get_model_valid(self):
        """Test get_model returns model data for valid model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            model_data = viz.get_model("dmd2")
            assert model_data is not None
            assert model_data.name == "dmd2"
            assert len(model_data.df) == 10

    def test_get_model_invalid(self):
        """Test get_model returns None for invalid model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz.get_model("unknown") is None

    def test_is_valid_model(self):
        """Test is_valid_model checks correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz.is_valid_model("dmd2") is True
            assert viz.is_valid_model("edm") is True
            assert viz.is_valid_model("unknown") is False

    def test_all_models_loaded_at_init(self):
        """Test that all models are preloaded at init (thread-safe pattern)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)
            create_model_dir(root, "edm", "edm-imagenet-64", num_samples=15)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Both models should have their data loaded
            dmd2_data = viz.get_model("dmd2")
            edm_data = viz.get_model("edm")

            assert dmd2_data is not None
            assert edm_data is not None
            assert len(dmd2_data.df) == 10
            assert len(edm_data.df) == 15


class TestGetPlotDataframe:
    """Test plot dataframe generation."""

    def test_basic_dataframe(self):
        """Test basic dataframe generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            plot_df = viz.get_plot_dataframe("dmd2")

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

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            plot_df = viz.get_plot_dataframe("dmd2", selected_idx=5)

            assert plot_df.loc[5, 'highlight'] == 'selected'
            assert sum(plot_df['highlight'] == 'selected') == 1

    def test_neighbor_highlights(self):
        """Test that neighbors are highlighted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            plot_df = viz.get_plot_dataframe(
                "dmd2",
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

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            model_data = viz.get_model("dmd2")
            # Get a class that exists in the data
            target_class = model_data.df['class_label'].iloc[0]
            expected_count = (model_data.df['class_label'] == target_class).sum()

            plot_df = viz.get_plot_dataframe("dmd2", highlighted_class=target_class)

            highlighted_count = (plot_df['highlight'] == 'class_highlight').sum()
            assert highlighted_count == expected_count


class TestClassOptions:
    """Test class options generation."""

    def test_get_class_options(self):
        """Test class options for dropdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            options = viz.get_class_options("dmd2")

            assert len(options) > 0
            # Options should be (label, value) tuples
            assert all(isinstance(opt, tuple) and len(opt) == 2 for opt in options)

    def test_class_options_sorted(self):
        """Test that class options are sorted by class ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=50)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            options = viz.get_class_options("dmd2")
            class_ids = [opt[1] for opt in options]

            assert class_ids == sorted(class_ids)


class TestGetImage:
    """Test image loading."""

    def test_get_image_success(self):
        """Test successful image loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            img = viz.get_image("dmd2", "images/imagenet_real/sample_000000.png")

            assert img is not None
            assert isinstance(img, np.ndarray)
            assert img.shape == (64, 64, 3)

    def test_get_image_not_found(self):
        """Test that missing image returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            img = viz.get_image("dmd2", "images/nonexistent.png")

            assert img is None


class TestCreateGradioApp:
    """Test Gradio app creation."""

    def test_app_creation(self):
        """Test that Gradio app is created successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
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

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            app = create_gradio_app(viz)

            assert app is not None
            assert len(viz.model_configs) == 2


class TestNearestNeighbors:
    """Test KNN functionality."""

    def test_knn_model_fitted_at_init(self):
        """Test that KNN model is fitted during initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=30)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            model_data = viz.get_model("dmd2")
            # KNN should be fitted automatically during init
            assert model_data.nn_model is not None

    def test_find_knn_neighbors(self):
        """Test finding KNN neighbors returns distances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=30)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            neighbors = viz.find_knn_neighbors("dmd2", 0, k=5)

            assert len(neighbors) == 5
            # Each neighbor is (idx, distance) tuple
            for idx, dist in neighbors:
                assert isinstance(idx, int)
                assert isinstance(dist, float)
                assert idx != 0  # Shouldn't include the query point
                assert dist >= 0

    def test_find_knn_neighbors_sorted_by_distance(self):
        """Test that neighbors are sorted by distance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=30)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            neighbors = viz.find_knn_neighbors("dmd2", 0, k=5)

            distances = [dist for _, dist in neighbors]
            assert distances == sorted(distances)

    def test_find_knn_neighbors_invalid_model(self):
        """Test that find_knn_neighbors returns empty for invalid model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            neighbors = viz.find_knn_neighbors("unknown", 0, k=5)
            assert neighbors == []

    def test_find_knn_neighbors_invalid_idx(self):
        """Test that find_knn_neighbors handles invalid index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Index out of range
            neighbors = viz.find_knn_neighbors("dmd2", 999, k=5)
            assert neighbors == []


class TestCreateUmapFigure:
    """Test Plotly figure creation."""

    def test_basic_figure(self):
        """Test basic figure creation returns Plotly Figure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure("dmd2")

            # Should be a Plotly Figure
            assert hasattr(fig, 'data')
            assert hasattr(fig, 'layout')
            # Should have at least the main scatter trace
            assert len(fig.data) >= 1
            # Main trace should have correct number of points
            assert len(fig.data[0].x) == 20

    def test_figure_with_selection(self):
        """Test figure with selected point has extra trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure("dmd2", selected_idx=5)

            # Should have main trace + selection trace
            assert len(fig.data) >= 2
            # Find selection trace
            sel_trace = next((t for t in fig.data if t.name == "selected"), None)
            assert sel_trace is not None
            assert len(sel_trace.x) == 1

    def test_figure_with_neighbors(self):
        """Test figure with neighbors has neighbor traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure(
                "dmd2",
                selected_idx=0,
                manual_neighbors=[1, 2],
                knn_neighbors=[3, 4, 5]
            )

            # Should have traces for selection, manual neighbors, and KNN neighbors
            trace_names = [t.name for t in fig.data]
            assert "selected" in trace_names
            assert "manual_neighbors" in trace_names
            assert "knn_neighbors" in trace_names

    def test_figure_invalid_model(self):
        """Test figure creation with invalid model returns empty figure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure("unknown")

            # Should still return a figure, just with no data
            assert hasattr(fig, 'data')
            assert hasattr(fig, 'layout')

    def test_figure_customdata_has_indices(self):
        """Test that figure customdata contains row indices for click handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure("dmd2")

            # Main trace should have customdata with indices
            main_trace = fig.data[0]
            assert main_trace.customdata is not None
            assert list(main_trace.customdata) == list(range(10))

    def test_figure_with_trajectory(self):
        """Test figure with denoising trajectory has trajectory traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Sample trajectory with (x, y, sigma) tuples
            trajectory = [
                (0.0, 0.0, 80.0),  # Start (high noise)
                (1.0, 1.0, 40.0),
                (2.0, 2.0, 10.0),
                (3.0, 3.0, 1.0),   # End (low noise)
            ]

            fig = viz.create_umap_figure(
                "dmd2",
                selected_idx=0,
                trajectory=trajectory
            )

            # Should have trajectory traces (indexed for multi-trajectory support)
            trace_names = [t.name for t in fig.data]
            assert "trajectory_line_0" in trace_names
            assert "trajectory_0" in trace_names
            assert "traj_start_0" in trace_names
            assert "traj_end_0" in trace_names

            # Trajectory trace should have correct number of points
            traj_trace = next(t for t in fig.data if t.name == "trajectory_0")
            assert len(traj_trace.x) == 4
            assert len(traj_trace.y) == 4

            # Start/end markers should have single point each
            start_trace = next(t for t in fig.data if t.name == "traj_start_0")
            end_trace = next(t for t in fig.data if t.name == "traj_end_0")
            assert len(start_trace.x) == 1
            assert len(end_trace.x) == 1
            assert start_trace.x[0] == 0.0  # First trajectory point
            assert end_trace.x[0] == 3.0    # Last trajectory point

    def test_figure_trajectory_empty(self):
        """Test that empty or single-point trajectory doesn't add traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Empty trajectory
            fig = viz.create_umap_figure("dmd2", trajectory=[])
            trace_names = [t.name for t in fig.data]
            assert "trajectory_0" not in trace_names

            # Single point (needs at least 2 for a path)
            fig = viz.create_umap_figure("dmd2", trajectory=[(0.0, 0.0, 80.0)])
            trace_names = [t.name for t in fig.data]
            assert "trajectory_0" not in trace_names


class TestMultiUserIsolation:
    """Test multi-user thread safety patterns."""

    def test_model_data_is_per_model(self):
        """Test that each model has isolated data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)
            create_model_dir(root, "edm", "edm-imagenet-64", num_samples=15)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            dmd2_data = viz.get_model("dmd2")
            edm_data = viz.get_model("edm")

            # Each model should have its own dataframe
            assert dmd2_data.df is not edm_data.df
            assert len(dmd2_data.df) != len(edm_data.df)

    def test_visualizer_has_no_current_model_attr(self):
        """Test that visualizer doesn't have mutable current_model attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Should not have current_model (that's session state now)
            assert not hasattr(viz, 'current_model')
            # Should have immutable default_model instead
            assert hasattr(viz, 'default_model')

    def test_methods_require_model_name(self):
        """Test that key methods require model_name parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # These methods should work with explicit model_name
            assert viz.get_plot_dataframe("dmd2") is not None
            assert viz.get_class_options("dmd2") is not None
            assert viz.create_umap_figure("dmd2") is not None


class TestLayerChoices:
    """Test layer choice retrieval."""

    def test_get_layer_choices_no_adapter(self):
        """Returns empty list when adapter not loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # No adapter loaded -> empty list
            model_data = viz.get_model("dmd2")
            model_data.adapter = None
            assert viz.get_layer_choices("dmd2") == []

    def test_get_layer_choices_with_adapter(self):
        """Returns hookable_layers from adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            mock_adapter = MagicMock()
            mock_adapter.hookable_layers = ["encoder_block_0", "midblock", "decoder_block_0"]
            viz.get_model("dmd2").adapter = mock_adapter

            choices = viz.get_layer_choices("dmd2")
            assert choices == ["encoder_block_0", "midblock", "decoder_block_0"]

    def test_get_layer_choices_invalid_model(self):
        """Returns empty list for unknown model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz.get_layer_choices("unknown") == []


class TestDefaultEmbeddingsBackup:
    """Test default embeddings backup and restore."""

    def test_default_fields_populated_at_init(self):
        """Default backup fields set during model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            md = viz.get_model("dmd2")
            assert md.default_df is not None
            assert len(md.default_df) == 10
            assert md.default_umap_params is not None
            assert md.default_nn_model is not None
            assert md.current_layer == "default"

    def test_restore_default_embeddings(self):
        """Restore swaps back original data after modification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            md = viz.get_model("dmd2")
            original_df = md.df.copy()

            # Simulate layer change by overwriting fields
            md.df = pd.DataFrame({"umap_x": [0], "umap_y": [0]})
            md.umap_params = {"layers": ["fake_layer"]}
            md.current_layer = "fake_layer"

            viz._restore_default_embeddings("dmd2")

            assert len(md.df) == 10
            assert md.current_layer == "default"
            assert md.df["umap_x"].tolist() == original_df["umap_x"].tolist()

    def test_restore_default_invalid_model(self):
        """Restore on invalid model is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Should not raise
            viz._restore_default_embeddings("unknown")


class TestLoadLayerCache:
    """Test layer cache loading from disk."""

    def test_cache_miss(self):
        """Returns False when no cache exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz._load_layer_cache("dmd2", "encoder_block_0") is False

    def test_cache_hit(self, monkeypatch):
        """Returns True and swaps data when cache exists."""
        monkeypatch.setenv("DIFFVIEWS_PCA_COMPONENTS", "0")
        n = 20  # must exceed n_neighbors=15
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            md = viz.get_model("dmd2")

            # Create fake cache files with enough samples for full UMAP fit
            cache_dir = md.data_dir / "embeddings" / "layer_cache"
            cache_dir.mkdir(parents=True)

            rng = np.random.RandomState(42)
            cached_df = pd.DataFrame({
                "sample_id": [f"s{i}" for i in range(n)],
                "umap_x": rng.randn(n),
                "umap_y": rng.randn(n),
            })
            cached_df.to_csv(cache_dir / "encoder_block_0.csv", index=False)

            cached_params = {"layers": ["encoder_block_0"], "n_neighbors": 15}
            with open(cache_dir / "encoder_block_0.json", "w") as f:
                json.dump(cached_params, f)

            np.save(cache_dir / "encoder_block_0.npy", rng.randn(n, 30))

            result = viz._load_layer_cache("dmd2", "encoder_block_0")
            assert result is True
            assert len(md.df) == n
            # Full UMAP refit from activations — reducer supports .transform()
            assert md.umap_reducer is not None
            assert hasattr(md.umap_reducer, 'transform')
            assert md.umap_scaler is not None
            assert md.activations.shape == (n, 30)
            assert md.current_layer == "encoder_block_0"
            assert md.umap_params["layers"] == ["encoder_block_0"]

    def test_cache_invalid_model(self):
        """Returns False for unknown model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            assert viz._load_layer_cache("unknown", "encoder_block_0") is False

    def test_cache_csv_only_loads_without_reducer(self):
        """CSV without pkl loads successfully (reducer is None, no refit without npy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            md = viz.get_model("dmd2")
            cache_dir = md.data_dir / "embeddings" / "layer_cache"
            cache_dir.mkdir(parents=True)

            # Only CSV, no PKL or NPY — loads but reducer stays None
            pd.DataFrame({"umap_x": [1.0], "umap_y": [2.0]}).to_csv(
                cache_dir / "midblock.csv", index=False
            )

            assert viz._load_layer_cache("dmd2", "midblock") is True
            assert md.umap_reducer is None
            assert md.current_layer == "midblock"


class TestExtractLayerActivations:
    """Test activation extraction (mocked GPU)."""

    def test_returns_none_no_metadata(self):
        """Returns None when metadata_df is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            md = viz.get_model("dmd2")
            assert md.metadata_df is None
            result = viz.extract_layer_activations("dmd2", "encoder_block_0")
            assert result is None

    def test_returns_none_no_adapter(self):
        """Returns None when adapter can't be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            metadata_df = pd.DataFrame({
                "image_path": ["images/imagenet_real/sample_000000.png"],
                "conditioning_sigma": [10.0],
                "class_label": [0],
            })

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, metadata_df)):
                viz = GradioVisualizer(data_dir=root)

            # No checkpoint -> adapter can't load
            result = viz.extract_layer_activations("dmd2", "encoder_block_0")
            assert result is None

    def test_returns_none_invalid_model(self):
        """Returns None for unknown model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            result = viz.extract_layer_activations("unknown", "encoder_block_0")
            assert result is None


class TestRecomputeLayerUmap:
    """Test UMAP recomputation orchestration."""

    def test_uses_cache_when_available(self):
        """Loads from cache instead of recomputing when cache exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Mock cache hit
            with patch.object(viz, '_load_layer_cache', return_value=True) as mock_cache:
                result = viz.recompute_layer_umap("dmd2", "encoder_block_0")

            assert result is True
            mock_cache.assert_called_once_with("dmd2", "encoder_block_0")

    def test_extracts_and_computes_on_cache_miss(self):
        """Full pipeline: extract -> UMAP -> cache -> swap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            metadata_df = pd.DataFrame({
                "sample_id": [f"s{i}" for i in range(5)],
                "image_path": [f"images/imagenet_real/sample_{i:06d}.png" for i in range(5)],
                "conditioning_sigma": [10.0] * 5,
                "class_label": list(range(5)),
            })

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, metadata_df)):
                viz = GradioVisualizer(data_dir=root)

            fake_activations = np.random.randn(5, 100).astype(np.float32)
            fake_embeddings = np.random.randn(5, 2).astype(np.float32)
            fake_reducer = MagicMock()
            fake_scaler = MagicMock()

            with patch.object(viz, '_load_layer_cache', return_value=False), \
                 patch.object(viz, 'extract_layer_activations', return_value=fake_activations) as mock_extract, \
                 patch('diffviews.visualization.app.GradioVisualizer.recompute_layer_umap') as _:
                # Call the real method manually to test orchestration
                pass

            # Test the cache-miss path with mocked extraction + UMAP
            with patch.object(viz, '_load_layer_cache', return_value=False), \
                 patch('diffviews.visualization.app._extract_layer_on_gpu', return_value=fake_activations), \
                 patch('diffviews.processing.umap.compute_umap', return_value=(fake_embeddings, fake_reducer, fake_scaler, None)), \
                 patch('diffviews.processing.umap.save_embeddings'):
                result = viz.recompute_layer_umap("dmd2", "encoder_block_0")

            assert result is True
            md = viz.get_model("dmd2")
            assert md.current_layer == "encoder_block_0"
            assert md.activations is fake_activations
            assert md.umap_reducer is fake_reducer
            assert md.umap_scaler is fake_scaler
            assert "umap_x" in md.df.columns

    def test_returns_false_on_extraction_failure(self):
        """Returns False when extraction fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            metadata_df = pd.DataFrame({
                "sample_id": ["s0"],
                "image_path": ["images/imagenet_real/sample_000000.png"],
                "conditioning_sigma": [10.0],
                "class_label": [0],
            })

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, metadata_df)):
                viz = GradioVisualizer(data_dir=root)

            with patch.object(viz, '_load_layer_cache', return_value=False), \
                 patch.object(viz, 'extract_layer_activations', return_value=None):
                result = viz.recompute_layer_umap("dmd2", "encoder_block_0")

            assert result is False

    def test_returns_false_invalid_model(self):
        """Returns False for unknown model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            result = viz.recompute_layer_umap("unknown", "encoder_block_0")
            assert result is False


class TestEvictLayerCache:
    """Tests for _evict_layer_cache disk budget enforcement."""

    def _make_viz(self, root):
        create_model_dir(root, "dmd2", "dmd2-imagenet-64")
        with patch.object(GradioVisualizer, '_load_activations', return_value=(None, None)):
            return GradioVisualizer(data_dir=root)

    def _make_layer_files(self, cache_dir, layer_name, size_mb=1, mtime_offset=0):
        """Create fake layer cache files with controlled size and mtime."""
        import time
        cache_dir.mkdir(parents=True, exist_ok=True)
        # .npy is the big one
        npy = cache_dir / f"{layer_name}.npy"
        npy.write_bytes(b"\x00" * int(size_mb * 1024 * 1024))
        csv = cache_dir / f"{layer_name}.csv"
        csv.write_text("umap_x,umap_y\n0,0\n")
        json_f = cache_dir / f"{layer_name}.json"
        json_f.write_text("{}")
        # Set mtime for ordering
        t = time.time() + mtime_offset
        for f in [npy, csv, json_f]:
            os.utime(f, (t, t))

    def test_no_eviction_under_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            viz = self._make_viz(root)
            cache_dir = root / "dmd2" / "embeddings" / "layer_cache"
            self._make_layer_files(cache_dir, "block_0", size_mb=1)

            viz._evict_layer_cache(cache_dir)
            assert (cache_dir / "block_0.npy").exists()

    def test_evicts_oldest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            viz = self._make_viz(root)
            # Set very small budget
            viz.LAYER_CACHE_MAX_BYTES = 3 * 1024 * 1024  # 3MB

            cache_dir = root / "dmd2" / "embeddings" / "layer_cache"
            self._make_layer_files(cache_dir, "old_layer", size_mb=1, mtime_offset=-100)
            self._make_layer_files(cache_dir, "new_layer", size_mb=1, mtime_offset=0)

            # Budget is 3MB, current usage ~2MB, requesting 300MB default
            viz._evict_layer_cache(cache_dir)

            # old_layer should be evicted first
            assert not (cache_dir / "old_layer.npy").exists()
            assert not (cache_dir / "old_layer.csv").exists()
            assert not (cache_dir / "old_layer.json").exists()

    def test_evicts_multiple_until_under_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            viz = self._make_viz(root)
            viz.LAYER_CACHE_MAX_BYTES = 5 * 1024 * 1024  # 5MB

            cache_dir = root / "dmd2" / "embeddings" / "layer_cache"
            self._make_layer_files(cache_dir, "oldest", size_mb=2, mtime_offset=-200)
            self._make_layer_files(cache_dir, "middle", size_mb=2, mtime_offset=-100)
            self._make_layer_files(cache_dir, "newest", size_mb=2, mtime_offset=0)

            # 6MB used, 5MB budget, need 300MB → must evict all three
            viz._evict_layer_cache(cache_dir, needed_bytes=2 * 1024 * 1024)

            # oldest evicted first, then middle
            assert not (cache_dir / "oldest.npy").exists()
            assert not (cache_dir / "middle.npy").exists()

    def test_noop_on_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            viz = self._make_viz(root)
            cache_dir = root / "dmd2" / "embeddings" / "layer_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Should not raise
            viz._evict_layer_cache(cache_dir)

    def test_noop_on_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            viz = self._make_viz(root)
            # Should not raise
            viz._evict_layer_cache(root / "nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

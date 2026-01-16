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

    def test_find_knn_neighbors(self):
        """Test finding KNN neighbors returns distances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=30)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            viz.fit_nearest_neighbors()
            neighbors = viz.find_knn_neighbors(0, k=5)

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

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            viz.fit_nearest_neighbors()
            neighbors = viz.find_knn_neighbors(0, k=5)

            distances = [dist for _, dist in neighbors]
            assert distances == sorted(distances)

    def test_find_knn_neighbors_no_model(self):
        """Test that find_knn_neighbors returns empty when no model fitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Don't fit the model
            neighbors = viz.find_knn_neighbors(0, k=5)
            assert neighbors == []

    def test_find_knn_neighbors_invalid_idx(self):
        """Test that find_knn_neighbors handles invalid index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            viz.fit_nearest_neighbors()
            # Index out of range
            neighbors = viz.find_knn_neighbors(999, k=5)
            assert neighbors == []


class TestCreateUmapFigure:
    """Test Plotly figure creation."""

    def test_basic_figure(self):
        """Test basic figure creation returns Plotly Figure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure()

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

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure(selected_idx=5)

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

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure(
                selected_idx=0,
                manual_neighbors=[1, 2],
                knn_neighbors=[3, 4, 5]
            )

            # Should have traces for selection, manual neighbors, and KNN neighbors
            trace_names = [t.name for t in fig.data]
            assert "selected" in trace_names
            assert "manual_neighbors" in trace_names
            assert "knn_neighbors" in trace_names

    def test_figure_empty_df(self):
        """Test figure creation with empty dataframe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=5)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            viz.df = pd.DataFrame()
            fig = viz.create_umap_figure()

            # Should still return a figure, just with no data
            assert hasattr(fig, 'data')
            assert hasattr(fig, 'layout')

    def test_figure_customdata_has_indices(self):
        """Test that figure customdata contains row indices for click handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            fig = viz.create_umap_figure()

            # Main trace should have customdata with indices
            main_trace = fig.data[0]
            assert main_trace.customdata is not None
            assert list(main_trace.customdata) == list(range(10))

    def test_figure_with_trajectory(self):
        """Test figure with denoising trajectory has trajectory traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=20)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Sample trajectory with (x, y, sigma) tuples
            trajectory = [
                (0.0, 0.0, 80.0),  # Start (high noise)
                (1.0, 1.0, 40.0),
                (2.0, 2.0, 10.0),
                (3.0, 3.0, 1.0),   # End (low noise)
            ]

            fig = viz.create_umap_figure(
                selected_idx=0,
                trajectory=trajectory
            )

            # Should have trajectory traces
            trace_names = [t.name for t in fig.data]
            assert "trajectory_line" in trace_names
            assert "trajectory" in trace_names
            assert "traj_start" in trace_names
            assert "traj_end" in trace_names

            # Trajectory trace should have correct number of points
            traj_trace = next(t for t in fig.data if t.name == "trajectory")
            assert len(traj_trace.x) == 4
            assert len(traj_trace.y) == 4

            # Start/end markers should have single point each
            start_trace = next(t for t in fig.data if t.name == "traj_start")
            end_trace = next(t for t in fig.data if t.name == "traj_end")
            assert len(start_trace.x) == 1
            assert len(end_trace.x) == 1
            assert start_trace.x[0] == 0.0  # First trajectory point
            assert end_trace.x[0] == 3.0    # Last trajectory point

    def test_figure_trajectory_empty(self):
        """Test that empty or single-point trajectory doesn't add traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64", num_samples=10)

            with patch.object(GradioVisualizer, 'load_activations_for_model', return_value=(None, None)):
                viz = GradioVisualizer(data_dir=root)

            # Empty trajectory
            fig = viz.create_umap_figure(trajectory=[])
            trace_names = [t.name for t in fig.data]
            assert "trajectory" not in trace_names

            # Single point (needs at least 2 for a path)
            fig = viz.create_umap_figure(trajectory=[(0.0, 0.0, 80.0)])
            trace_names = [t.name for t in fig.data]
            assert "trajectory" not in trace_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

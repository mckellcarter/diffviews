"""
Unit tests for diffviews.visualization.app model switching
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from diffviews.visualization.app import DMD2Visualizer


class TestModelConfigs:
    """Test model configuration setup."""

    def test_model_configs_initialized(self):
        """Test that model configs are set up correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "dmd2"
            edm_dir = Path(tmpdir) / "edm"
            data_dir.mkdir()
            edm_dir.mkdir()

            # Create minimal structure
            (data_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (data_dir / "metadata" / "imagenet_real").mkdir(parents=True)
            (edm_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (edm_dir / "metadata" / "imagenet_real").mkdir(parents=True)

            # Mock load_data to avoid actual file loading
            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(
                            data_dir=data_dir,
                            embeddings_path=None,
                            edm_data_dir=edm_dir,
                            edm_embeddings_path="edm_embeddings.csv"
                        )

            assert 'dmd2' in viz.model_configs
            assert 'edm' in viz.model_configs
            assert viz.model_configs['dmd2']['data_dir'] == data_dir
            assert viz.model_configs['edm']['data_dir'] == edm_dir
            assert viz.current_model == 'dmd2'

    def test_default_model_is_dmd2(self):
        """Test that default model is dmd2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (data_dir / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=data_dir)

            assert viz.current_model == 'dmd2'

    def test_edm_not_configured_without_paths(self):
        """Test EDM config is None without paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (data_dir / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=data_dir)

            assert viz.model_configs['edm']['data_dir'] is None
            assert viz.model_configs['edm']['embeddings_path'] is None


class TestSwitchModel:
    """Test model switching functionality."""

    def test_switch_model_unknown(self):
        """Test switching to unknown model fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (data_dir / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=data_dir)

            result = viz.switch_model('unknown_model')
            assert result is False
            assert viz.current_model == 'dmd2'

    def test_switch_model_unconfigured(self):
        """Test switching to unconfigured model fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (data_dir / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=data_dir)
                        # EDM not configured

            result = viz.switch_model('edm')
            assert result is False
            assert viz.current_model == 'dmd2'

    def test_switch_model_resets_state(self):
        """Test that switch_model resets adapter and model state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "dmd2"
            edm_dir = Path(tmpdir) / "edm"
            data_dir.mkdir()
            edm_dir.mkdir()

            for d in [data_dir, edm_dir]:
                (d / "activations" / "imagenet_real").mkdir(parents=True)
                (d / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(
                            data_dir=data_dir,
                            embeddings_path="test.csv",
                            edm_data_dir=edm_dir,
                            edm_embeddings_path="edm.csv"
                        )

            # Set some state
            viz.adapter = MagicMock()
            viz.layer_shapes = {'test': (1, 2, 3)}
            viz.umap_reducer = MagicMock()
            viz.df = pd.DataFrame({'x': [1, 2, 3]})

            # Switch model
            with patch.object(viz, 'load_data'):
                with patch.object(viz, 'fit_nearest_neighbors'):
                    result = viz.switch_model('edm')

            assert result is True
            assert viz.current_model == 'edm'
            assert viz.adapter is None
            assert viz.layer_shapes == {}
            assert viz.umap_reducer is None

    def test_switch_model_updates_paths(self):
        """Test that switch_model updates data_dir and embeddings_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "dmd2"
            edm_dir = Path(tmpdir) / "edm"
            data_dir.mkdir()
            edm_dir.mkdir()

            for d in [data_dir, edm_dir]:
                (d / "activations" / "imagenet_real").mkdir(parents=True)
                (d / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(
                            data_dir=data_dir,
                            embeddings_path="dmd2.csv",
                            edm_data_dir=edm_dir,
                            edm_embeddings_path="edm.csv"
                        )

            assert viz.data_dir == data_dir
            assert viz.embeddings_path == "dmd2.csv"

            with patch.object(viz, 'load_data'):
                with patch.object(viz, 'fit_nearest_neighbors'):
                    viz.switch_model('edm')

            assert viz.data_dir == edm_dir
            assert viz.embeddings_path == "edm.csv"
            assert viz.adapter_name == 'edm-imagenet-64'


class TestClassNames:
    """Test class name handling."""

    def test_get_class_name_known(self):
        """Test getting known class name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (data_dir / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=data_dir)

            viz.class_labels = {88: 'macaw', 207: 'golden_retriever'}

            assert viz.get_class_name(88) == 'macaw'
            assert viz.get_class_name(207) == 'golden_retriever'

    def test_get_class_name_unknown(self):
        """Test getting unknown class name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "activations" / "imagenet_real").mkdir(parents=True)
            (data_dir / "metadata" / "imagenet_real").mkdir(parents=True)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=data_dir)

            viz.class_labels = {}

            result = viz.get_class_name(999)
            assert 'Unknown' in result or '999' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

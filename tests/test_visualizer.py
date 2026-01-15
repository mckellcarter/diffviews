"""
Unit tests for diffviews.visualization.app model switching
"""

import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from diffviews.visualization.app import DMD2Visualizer


def create_model_dir(root: Path, model_name: str, adapter_name: str):
    """Helper to create minimal model directory structure."""
    model_dir = root / model_name
    model_dir.mkdir()
    (model_dir / "activations" / "imagenet_real").mkdir(parents=True)
    (model_dir / "metadata" / "imagenet_real").mkdir(parents=True)
    (model_dir / "embeddings").mkdir(parents=True)
    # Create config.json
    with open(model_dir / "config.json", "w") as f:
        json.dump({"adapter": adapter_name}, f)
    # Create empty embeddings CSV with required columns
    pd.DataFrame(columns=["sample_id", "umap_x", "umap_y"]).to_csv(
        model_dir / "embeddings" / "demo_embeddings.csv", index=False
    )
    return model_dir


class TestModelConfigs:
    """Test model configuration setup."""

    def test_model_configs_initialized(self):
        """Test that model configs are set up correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dmd2_dir = create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            edm_dir = create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            assert 'dmd2' in viz.model_configs
            assert 'edm' in viz.model_configs
            assert viz.model_configs['dmd2']['data_dir'] == dmd2_dir
            assert viz.model_configs['edm']['data_dir'] == edm_dir
            assert viz.current_model == 'dmd2'

    def test_default_model_is_dmd2(self):
        """Test that default model is dmd2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            assert viz.current_model == 'dmd2'

    def test_only_configured_models_discovered(self):
        """Test that only models with config.json + embeddings are discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            # Create incomplete edm dir (no embeddings)
            (root / "edm" / "config.json").parent.mkdir(parents=True)
            with open(root / "edm" / "config.json", "w") as f:
                json.dump({"adapter": "edm-imagenet-64"}, f)

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            assert 'dmd2' in viz.model_configs
            assert 'edm' not in viz.model_configs


class TestSwitchModel:
    """Test model switching functionality."""

    def test_switch_model_unknown(self):
        """Test switching to unknown model fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            result = viz.switch_model('unknown_model')
            assert result is False
            assert viz.current_model == 'dmd2'

    def test_switch_model_unconfigured(self):
        """Test switching to unconfigured model fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            result = viz.switch_model('edm')
            assert result is False
            assert viz.current_model == 'dmd2'

    def test_switch_model_resets_state(self):
        """Test that switch_model resets adapter and model state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

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
        """Test that switch_model updates data_dir and adapter_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dmd2_dir = create_model_dir(root, "dmd2", "dmd2-imagenet-64")
            edm_dir = create_model_dir(root, "edm", "edm-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            assert viz.data_dir == dmd2_dir
            assert viz.adapter_name == 'dmd2-imagenet-64'

            with patch.object(viz, 'load_data'):
                with patch.object(viz, 'fit_nearest_neighbors'):
                    viz.switch_model('edm')

            assert viz.data_dir == edm_dir
            assert viz.adapter_name == 'edm-imagenet-64'


class TestClassNames:
    """Test class name handling."""

    def test_get_class_name_known(self):
        """Test getting known class name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            viz.class_labels = {88: 'macaw', 207: 'golden_retriever'}

            assert viz.get_class_name(88) == 'macaw'
            assert viz.get_class_name(207) == 'golden_retriever'

    def test_get_class_name_unknown(self):
        """Test getting unknown class name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_model_dir(root, "dmd2", "dmd2-imagenet-64")

            with patch.object(DMD2Visualizer, 'load_data'):
                with patch.object(DMD2Visualizer, 'build_layout'):
                    with patch.object(DMD2Visualizer, 'register_callbacks'):
                        viz = DMD2Visualizer(data_dir=root)

            viz.class_labels = {}

            result = viz.get_class_name(999)
            assert 'Unknown' in result or '999' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

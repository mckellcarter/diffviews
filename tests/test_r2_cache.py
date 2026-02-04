"""Tests for R2LayerCache with mocked boto3."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from diffviews.data.r2_cache import R2LayerCache, LAYER_CACHE_EXTENSIONS


@pytest.fixture
def r2_env(monkeypatch):
    """Set R2 env vars for testing."""
    monkeypatch.setenv("R2_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("R2_BUCKET_NAME", "test-bucket")


@pytest.fixture
def mock_boto3():
    """Mock boto3 imported inside R2LayerCache.__init__."""
    mock_module = MagicMock()
    client = MagicMock()
    mock_module.client.return_value = client
    with patch.dict("sys.modules", {"boto3": mock_module}):
        yield client


class TestR2Init:
    def test_disabled_without_credentials(self):
        """R2 disabled when env vars missing."""
        with patch.dict("os.environ", {}, clear=True):
            cache = R2LayerCache()
        assert cache.enabled is False

    def test_disabled_partial_credentials(self, monkeypatch):
        """R2 disabled with only some env vars."""
        monkeypatch.setenv("R2_ACCOUNT_ID", "test")
        # Missing access key and secret
        cache = R2LayerCache()
        assert cache.enabled is False

    def test_enabled_with_credentials(self, r2_env, mock_boto3):
        """R2 enabled with all env vars + boto3."""
        cache = R2LayerCache()
        assert cache.enabled is True

    def test_disabled_without_boto3(self, r2_env):
        """R2 disabled when boto3 not importable."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            cache = R2LayerCache()
        assert cache.enabled is False

    def test_custom_bucket(self, r2_env, mock_boto3):
        """Custom bucket name overrides env."""
        cache = R2LayerCache(bucket="custom-bucket")
        assert cache._bucket == "custom-bucket"

    def test_default_bucket(self, r2_env, mock_boto3):
        """Uses R2_BUCKET_NAME env var."""
        cache = R2LayerCache()
        assert cache._bucket == "test-bucket"


class TestKeyConstruction:
    def test_key_format(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        assert cache._key("dmd2", "encoder_block_0", ".csv") == \
            "data/dmd2/layer_cache/encoder_block_0.csv"
        assert cache._key("edm", "midblock", ".npy") == \
            "data/edm/layer_cache/midblock.npy"


class TestLayerExists:
    def test_exists_true(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        mock_boto3.head_object.return_value = {}
        assert cache.layer_exists("dmd2", "encoder_block_0") is True
        mock_boto3.head_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="data/dmd2/layer_cache/encoder_block_0.csv",
        )

    def test_exists_false_on_error(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        mock_boto3.head_object.side_effect = Exception("404")
        assert cache.layer_exists("dmd2", "encoder_block_0") is False

    def test_exists_false_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            cache = R2LayerCache()
        assert cache.layer_exists("dmd2", "encoder_block_0") is False


class TestDownloadLayer:
    def test_downloads_all_extensions(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            # Create fake files that download_file would create
            for ext in LAYER_CACHE_EXTENSIONS:
                (local_dir / f"encoder_block_0{ext}").write_text("x")

            result = cache.download_layer("dmd2", "encoder_block_0", local_dir)

        assert result is True
        assert mock_boto3.download_file.call_count == 3

    def test_returns_false_on_csv_failure(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        mock_boto3.download_file.side_effect = Exception("network error")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = cache.download_layer("dmd2", "encoder_block_0", Path(tmpdir))
        assert result is False

    def test_returns_false_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            cache = R2LayerCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = cache.download_layer("dmd2", "encoder_block_0", Path(tmpdir))
        assert result is False

    def test_creates_local_dir(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            # Will fail on download but should create dir
            mock_boto3.download_file.side_effect = Exception("err")
            cache.download_layer("dmd2", "encoder_block_0", nested)
            assert nested.exists()


class TestUploadLayer:
    def test_uploads_existing_files(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            for ext in LAYER_CACHE_EXTENSIONS:
                (local_dir / f"layer{ext}").write_text("data")

            result = cache.upload_layer("dmd2", "layer", local_dir)

        assert result is True
        assert mock_boto3.upload_file.call_count == 3

    def test_skips_missing_files(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            # Only create .csv
            (local_dir / "layer.csv").write_text("data")

            result = cache.upload_layer("dmd2", "layer", local_dir)

        assert result is True
        assert mock_boto3.upload_file.call_count == 1

    def test_returns_false_on_csv_upload_failure(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        mock_boto3.upload_file.side_effect = Exception("upload failed")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            (local_dir / "layer.csv").write_text("data")
            result = cache.upload_layer("dmd2", "layer", local_dir)
        assert result is False

    def test_returns_false_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            cache = R2LayerCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = cache.upload_layer("dmd2", "layer", Path(tmpdir))
        assert result is False


class TestUploadLayerAsync:
    def test_fires_background_thread(self, r2_env, mock_boto3):
        cache = R2LayerCache()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            (local_dir / "layer.csv").write_text("data")

            with patch.object(cache, "upload_layer") as mock_upload:
                cache.upload_layer_async("dmd2", "layer", local_dir)
                # Give thread a moment
                import time
                time.sleep(0.1)
                mock_upload.assert_called_once_with("dmd2", "layer", local_dir)

    def test_noop_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            cache = R2LayerCache()
        # Should not raise
        cache.upload_layer_async("dmd2", "layer", Path("/tmp"))

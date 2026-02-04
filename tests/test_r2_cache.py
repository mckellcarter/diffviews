"""Tests for R2LayerCache and R2DataStore with mocked boto3."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from diffviews.data.r2_cache import R2LayerCache, R2DataStore, LAYER_CACHE_EXTENSIONS


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


# ──────────────────────────────────────────────
# R2DataStore tests
# ──────────────────────────────────────────────

class TestR2DataStoreInit:
    def test_disabled_without_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            store = R2DataStore()
        assert store.enabled is False

    def test_enabled_with_credentials(self, r2_env, mock_boto3):
        store = R2DataStore()
        assert store.enabled is True


class TestR2DataStoreListObjects:
    def test_lists_keys(self, r2_env, mock_boto3):
        store = R2DataStore()
        paginator = MagicMock()
        mock_boto3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "data/dmd2/config.json"}, {"Key": "data/dmd2/embeddings/demo.csv"}]},
        ]
        keys = store.list_objects("data/dmd2/")
        assert keys == ["data/dmd2/config.json", "data/dmd2/embeddings/demo.csv"]

    def test_empty_on_error(self, r2_env, mock_boto3):
        store = R2DataStore()
        mock_boto3.get_paginator.side_effect = Exception("fail")
        assert store.list_objects("data/") == []

    def test_empty_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            store = R2DataStore()
        assert store.list_objects("data/") == []


class TestR2DataStoreFileExists:
    def test_exists_true(self, r2_env, mock_boto3):
        store = R2DataStore()
        mock_boto3.head_object.return_value = {}
        assert store.file_exists("data/dmd2/config.json") is True

    def test_exists_false(self, r2_env, mock_boto3):
        store = R2DataStore()
        mock_boto3.head_object.side_effect = Exception("404")
        assert store.file_exists("data/dmd2/config.json") is False

    def test_exists_false_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            store = R2DataStore()
        assert store.file_exists("data/dmd2/config.json") is False


class TestR2DataStoreDownloadFile:
    def test_downloads_single_file(self, r2_env, mock_boto3):
        store = R2DataStore()
        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / "sub" / "file.json"
            # download_file creates parent dirs
            result = store.download_file("data/dmd2/config.json", local)
        assert result is True
        mock_boto3.download_file.assert_called_once()

    def test_returns_false_on_error(self, r2_env, mock_boto3):
        store = R2DataStore()
        mock_boto3.download_file.side_effect = Exception("err")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = store.download_file("data/x", Path(tmpdir) / "x")
        assert result is False

    def test_returns_false_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            store = R2DataStore()
        assert store.download_file("k", Path("/tmp/x")) is False


class TestR2DataStoreDownloadPrefix:
    def test_downloads_all_listed_objects(self, r2_env, mock_boto3):
        store = R2DataStore()
        paginator = MagicMock()
        mock_boto3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [
                {"Key": "data/dmd2/config.json"},
                {"Key": "data/dmd2/embeddings/demo.csv"},
            ]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            # Make download_file create files so stat() works
            def fake_download(bucket, key, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("x")
            mock_boto3.download_file.side_effect = fake_download

            count = store.download_prefix("data/", local_dir)

        assert count == 2
        assert mock_boto3.download_file.call_count == 2

    def test_skips_existing_files(self, r2_env, mock_boto3):
        store = R2DataStore()
        paginator = MagicMock()
        mock_boto3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "data/dmd2/config.json"}]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            # Pre-create file
            (local_dir / "dmd2").mkdir()
            (local_dir / "dmd2" / "config.json").write_text("existing")

            count = store.download_prefix("data/", local_dir)

        assert count == 0
        mock_boto3.download_file.assert_not_called()

    def test_returns_zero_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            store = R2DataStore()
        assert store.download_prefix("data/", Path("/tmp")) == 0


class TestR2DataStoreDownloadModelData:
    def test_downloads_model_and_shared(self, r2_env, mock_boto3):
        store = R2DataStore()
        paginator = MagicMock()
        mock_boto3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "data/dmd2/config.json"}]},
        ]

        def fake_download(bucket, key, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("x")
        mock_boto3.download_file.side_effect = fake_download

        with tempfile.TemporaryDirectory() as tmpdir:
            result = store.download_model_data("dmd2", Path(tmpdir))

        assert result is True
        # 1 model file + 2 shared files
        assert mock_boto3.download_file.call_count == 3

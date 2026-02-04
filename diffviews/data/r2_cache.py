"""
Cloudflare R2 storage for data hosting and UMAP layer cache.

Uses boto3 S3-compatible API. Gracefully degrades when credentials
are missing or R2 is unreachable — never on the critical path.

Classes:
    R2DataStore  — bulk model data download (checkpoints, activations, etc.)
    R2LayerCache — layer-specific UMAP cache (csv/json/npy only)

Env vars (set as HF Spaces Secrets or Modal secrets):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME (default: diffviews)
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


# Extensions we cache on R2 (portable only — no .pkl)
LAYER_CACHE_EXTENSIONS = (".csv", ".json", ".npy")


def _make_r2_client(bucket: Optional[str] = None):
    """Create boto3 S3 client for R2. Returns (client, bucket, enabled)."""
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = bucket or os.environ.get("R2_BUCKET_NAME", "diffviews")

    if not all([account_id, access_key, secret_key]):
        return None, bucket_name, False

    try:
        import boto3
        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
        return client, bucket_name, True
    except ImportError:
        return None, bucket_name, False
    except Exception:
        return None, bucket_name, False


class R2DataStore:
    """Bulk data download from R2 for model data hosting."""

    def __init__(self, bucket: Optional[str] = None):
        self._client, self._bucket, self._enabled = _make_r2_client(bucket)
        if self._enabled:
            print(f"[R2] DataStore connected to '{self._bucket}'")
        else:
            print("[R2] DataStore disabled (credentials or boto3 missing)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys under prefix."""
        if not self._enabled:
            return []
        keys = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except Exception as e:
            print(f"[R2] list_objects failed for {prefix}: {e}")
        return keys

    def file_exists(self, key: str) -> bool:
        """HEAD check on a single key."""
        if not self._enabled:
            return False
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def download_file(self, key: str, local_path: Path) -> bool:
        """Download single file from R2."""
        if not self._enabled:
            return False
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self._bucket, key, str(local_path))
            return True
        except Exception as e:
            print(f"[R2] Download failed: {key}: {e}")
            return False

    def download_prefix(
        self, prefix: str, local_dir: Path,
        exclude_dirs: set[str] | None = None,
        max_workers: int = 8,
    ) -> int:
        """Download all objects under prefix to local_dir, preserving structure.

        Keys like 'data/dmd2/config.json' with prefix='data/' download to
        local_dir/dmd2/config.json.

        Args:
            exclude_dirs: Set of directory names to skip (e.g. {"layer_cache"}).
            max_workers: Concurrent download threads.

        Returns count of files downloaded.
        """
        if not self._enabled:
            return 0

        keys = self.list_objects(prefix)
        if not keys:
            print(f"[R2] No objects found under {prefix}")
            return 0

        local_dir = Path(local_dir)

        # Filter out excluded dirs and already-existing files
        to_download = []
        for key in keys:
            rel = key[len(prefix):]
            if exclude_dirs:
                parts = Path(rel).parts
                if any(p in exclude_dirs for p in parts):
                    continue
            local_path = local_dir / rel
            if local_path.exists():
                continue
            to_download.append((key, local_path))

        if not to_download:
            return 0

        print(f"[R2] Downloading {len(to_download)} files ({max_workers} threads)...")
        downloaded = 0
        failed = 0

        def _download_one(key_path):
            key, local_path = key_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, key, str(local_path))
            return key, local_path

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_one, kp): kp for kp in to_download}
            for future in as_completed(futures):
                key, local_path = futures[future]
                try:
                    future.result()
                    downloaded += 1
                    if downloaded % 50 == 0 or downloaded == len(to_download):
                        print(f"[R2] Progress: {downloaded}/{len(to_download)} files")
                except Exception as e:
                    failed += 1
                    print(f"[R2] Failed {key}: {e}")

        if failed:
            print(f"[R2] Done: {downloaded} downloaded, {failed} failed")
        return downloaded

    def download_model_data(self, model: str, local_dir: Path) -> bool:
        """Download all data for a model from R2.

        Downloads data/{model}/* to local_dir/{model}/*.
        Excludes layer_cache/ (lazy-loaded on demand via R2LayerCache).
        Returns True if any files were downloaded or already exist.
        """
        prefix = f"data/{model}/"
        count = self.download_prefix(prefix, local_dir, exclude_dirs={"layer_cache"})
        # Also download root-level shared files
        for shared in ["imagenet_standard_class_index.json", "imagenet64_class_labels.json"]:
            key = f"data/{shared}"
            local_path = local_dir / shared
            if not local_path.exists():
                self.download_file(key, local_path)
        return count > 0 or (local_dir / model).exists()


class R2LayerCache:
    """S3-compatible client for Cloudflare R2 layer cache."""

    def __init__(self, bucket: Optional[str] = None):
        self._client, self._bucket, self._enabled = _make_r2_client(bucket)
        if self._enabled:
            print(f"[R2] LayerCache connected to '{self._bucket}'")
        else:
            print("[R2] LayerCache disabled (credentials or boto3 missing)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _key(self, model: str, layer: str, ext: str) -> str:
        """R2 object key for a layer cache file."""
        return f"data/{model}/layer_cache/{layer}{ext}"

    def layer_exists(self, model: str, layer: str) -> bool:
        """Check if layer cache exists on R2 (HEAD on .csv key)."""
        if not self._enabled:
            return False
        try:
            self._client.head_object(
                Bucket=self._bucket,
                Key=self._key(model, layer, ".csv"),
            )
            return True
        except Exception:
            return False

    def download_layer(self, model: str, layer: str, local_dir: Path) -> bool:
        """Download layer cache files (.csv, .json, .npy) to local_dir.

        Returns True if at least the .csv was downloaded successfully.
        """
        if not self._enabled:
            return False

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        got_csv = False

        for ext in LAYER_CACHE_EXTENSIONS:
            key = self._key(model, layer, ext)
            local_path = local_dir / f"{layer}{ext}"
            try:
                self._client.download_file(self._bucket, key, str(local_path))
                print(f"[R2] Downloaded {key} ({local_path.stat().st_size / 1e6:.1f}MB)")
                if ext == ".csv":
                    got_csv = True
            except Exception as e:
                if ext == ".csv":
                    print(f"[R2] Download failed for {key}: {e}")
                # .json/.npy missing is non-fatal

        return got_csv

    def upload_layer(self, model: str, layer: str, local_dir: Path) -> bool:
        """Upload layer cache files (.csv, .json, .npy) from local_dir to R2.

        Returns True if at least the .csv was uploaded successfully.
        """
        if not self._enabled:
            return False

        local_dir = Path(local_dir)
        uploaded_csv = False

        for ext in LAYER_CACHE_EXTENSIONS:
            local_path = local_dir / f"{layer}{ext}"
            if not local_path.exists():
                continue
            key = self._key(model, layer, ext)
            try:
                self._client.upload_file(str(local_path), self._bucket, key)
                print(f"[R2] Uploaded {key} ({local_path.stat().st_size / 1e6:.1f}MB)")
                if ext == ".csv":
                    uploaded_csv = True
            except Exception as e:
                print(f"[R2] Upload failed for {key}: {e}")

        return uploaded_csv

    def upload_layer_async(self, model: str, layer: str, local_dir: Path):
        """Fire-and-forget background upload. Does not block caller."""
        if not self._enabled:
            return
        t = threading.Thread(
            target=self.upload_layer,
            args=(model, layer, local_dir),
            daemon=True,
        )
        t.start()

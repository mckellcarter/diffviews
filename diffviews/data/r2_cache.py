"""
Cloudflare R2 cache layer for UMAP embeddings and activations.

Uses boto3 S3-compatible API. Gracefully degrades when credentials
are missing or R2 is unreachable — never on the critical path.

Env vars (set as HF Spaces Secrets or Modal secrets):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME (default: diffviews)
"""

import os
import threading
from pathlib import Path
from typing import Optional


# Extensions we cache on R2 (portable only — no .pkl)
LAYER_CACHE_EXTENSIONS = (".csv", ".json", ".npy")


class R2LayerCache:
    """S3-compatible client for Cloudflare R2 layer cache."""

    def __init__(self, bucket: Optional[str] = None):
        self._enabled = False
        self._client = None

        account_id = os.environ.get("R2_ACCOUNT_ID")
        access_key = os.environ.get("R2_ACCESS_KEY_ID")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self._bucket = bucket or os.environ.get("R2_BUCKET_NAME", "diffviews")

        if not all([account_id, access_key, secret_key]):
            print("[R2] Credentials not configured, R2 cache disabled")
            return

        try:
            import boto3
            self._client = boto3.client(
                "s3",
                endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="auto",
            )
            self._enabled = True
            print(f"[R2] Connected to bucket '{self._bucket}'")
        except ImportError:
            print("[R2] boto3 not installed, R2 cache disabled")
        except Exception as e:
            print(f"[R2] Init failed: {e}")

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

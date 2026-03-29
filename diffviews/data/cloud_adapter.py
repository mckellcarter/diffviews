"""
Cloud-backed ImageNet image fetching via CF Worker API.

Fetches ImageNet 64x64 images on-demand from yodal-items-api worker.
Gracefully degrades when credentials are missing.

Env vars (set as Modal secrets or HF Spaces secrets):
    YODAL_ITEMS_API_KEY  — API key for yodal-items-api worker
    YODAL_ITEMS_API_URL  — (optional) override base URL

Local dev: use keyring to store the key:
    keyring set yodal items_api_key
"""

import os
from typing import Optional
import numpy as np

try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False


class ImageNetCloudAdapter:
    """
    Fetch ImageNet images from CF Worker API.

    Images are returned as (H, W, 3) uint8 numpy arrays suitable for display.
    """

    DEFAULT_URL = "https://yodal-items-api.mckellcarter.workers.dev"
    IMAGENET_64_SHAPE = (64, 64, 3)
    IMAGENET_64_BYTES = 64 * 64 * 3  # 12288

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Args:
            base_url: API base URL (default: from env or DEFAULT_URL)
            api_key: API key (default: from env or keyring)
            timeout: Request timeout in seconds
        """
        self._base_url = (
            base_url
            or os.environ.get("YODAL_ITEMS_API_URL")
            or self.DEFAULT_URL
        ).rstrip("/")

        self._api_key = api_key or self._get_api_key()
        self._timeout = timeout
        self._enabled = self._api_key is not None

        if self._enabled:
            print(f"[CloudAdapter] ImageNet API enabled: {self._base_url}")
        else:
            print("[CloudAdapter] ImageNet API disabled (no API key found)")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or keyring."""
        key = os.environ.get("YODAL_ITEMS_API_KEY")
        if key:
            return key
        if HAS_KEYRING:
            # Try multiple keyring patterns
            patterns = [
                ("YODAL_ITEMS_API_KEY", "mckell"),
                ("yodal", "items_api_key"),
                ("YODAL_ITEMS_API_KEY", "default"),
            ]
            for service, username in patterns:
                try:
                    val = keyring.get_password(service, username)
                    if val:
                        return val
                except Exception:
                    pass
        return None

    @property
    def enabled(self) -> bool:
        """Whether cloud adapter is configured and available."""
        return self._enabled

    def _request(self, path: str) -> bytes:
        """Make HTTP request to API."""
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError

        url = f"{self._base_url}{path}"
        headers = {
            "User-Agent": "diffviews/1.0",  # CF blocks default Python UA
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                return resp.read()
        except HTTPError as e:
            if e.code == 404:
                raise KeyError(f"Item not found: {path}") from e
            raise

    def get_image_bytes(self, item_id: str) -> bytes:
        """
        Fetch raw image bytes from API.

        Args:
            item_id: ImageNet index as string (e.g., "12345")

        Returns:
            Raw pixel bytes (12288 for 64x64 RGB)
        """
        return self._request(f"/v1/imagenet/item/{item_id}")

    def get_image(self, item_id: str) -> np.ndarray:
        """
        Fetch image as numpy array.

        Args:
            item_id: ImageNet index as string

        Returns:
            (64, 64, 3) uint8 numpy array (RGB, HWC format)
        """
        data = self.get_image_bytes(item_id)
        if len(data) != self.IMAGENET_64_BYTES:
            raise ValueError(
                f"Unexpected byte length: {len(data)}, expected {self.IMAGENET_64_BYTES}"
            )
        # Raw bytes are CHW (channel-first), convert to HWC for display
        chw = np.frombuffer(data, dtype=np.uint8).reshape(3, 64, 64)
        return chw.transpose(1, 2, 0)

    def get_image_from_sample_id(self, sample_id: str) -> Optional[np.ndarray]:
        """
        Fetch image using sample_id format (e.g., "sample_000009").

        Args:
            sample_id: Sample identifier like "sample_XXXXXX"

        Returns:
            (64, 64, 3) uint8 numpy array or None if fetch fails
        """
        try:
            # Extract numeric ID from sample_XXXXXX format
            if sample_id.startswith("sample_"):
                item_id = str(int(sample_id.split("_")[1]))
            else:
                item_id = sample_id
            return self.get_image(item_id)
        except (KeyError, ValueError, IndexError) as e:
            print(f"Cloud fetch failed for {sample_id}: {e}")
            return None


# Module-level singleton for convenience
_cloud_adapter: Optional[ImageNetCloudAdapter] = None


def get_cloud_adapter() -> ImageNetCloudAdapter:
    """Get or create the global cloud adapter instance."""
    global _cloud_adapter
    if _cloud_adapter is None:
        _cloud_adapter = ImageNetCloudAdapter()
    return _cloud_adapter


def cloud_enabled() -> bool:
    """Check if cloud adapter is available."""
    return get_cloud_adapter().enabled

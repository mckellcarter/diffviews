#!/usr/bin/env python3
"""Seed R2 bucket with local data/ directory contents.

Uploads model data (activations, embeddings, images, metadata, checkpoints)
to Cloudflare R2. Skips non-portable artifacts (.pkl in embeddings/,
.npz old format, intermediates/, .DS_Store).

Usage:
    python scripts/seed_r2.py                    # dry-run (default)
    python scripts/seed_r2.py --execute          # actually upload
    python scripts/seed_r2.py --data-dir /path   # custom data dir

Requires env vars: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
Optional: R2_BUCKET_NAME (default: diffviews)
"""

import argparse
import os
import sys
from pathlib import Path

# Skip patterns (relative to data/)
SKIP_DIRS = {"intermediates", "layer_cache", "__pycache__", ".git"}
SKIP_FILES = {".DS_Store", ".gitkeep"}
SKIP_EXTENSIONS_IN = {
    # .pkl in embeddings/ not portable (numba JIT)
    "embeddings": {".pkl"},
}
# .npz superseded by .npy
SKIP_EXTENSIONS_GLOBAL = {".npz"}


def should_skip(rel_path: Path) -> str | None:
    """Return skip reason or None if file should be uploaded."""
    for part in rel_path.parts:
        if part in SKIP_DIRS:
            return f"skip dir: {part}"

    if rel_path.name in SKIP_FILES:
        return f"skip file: {rel_path.name}"

    if rel_path.suffix in SKIP_EXTENSIONS_GLOBAL:
        return f"skip ext: {rel_path.suffix}"

    # Context-sensitive extension skips
    for dir_name, exts in SKIP_EXTENSIONS_IN.items():
        if dir_name in rel_path.parts and rel_path.suffix in exts:
            return f"skip {rel_path.suffix} in {dir_name}/"

    return None


def collect_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Walk data_dir, return (local_path, r2_key) pairs."""
    files = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(data_dir)
        reason = should_skip(rel)
        if reason:
            print(f"  SKIP {rel}  ({reason})")
            continue
        r2_key = f"data/{rel}"
        files.append((path, r2_key))
    return files


def main():
    parser = argparse.ArgumentParser(description="Seed R2 with local data/")
    parser.add_argument("--data-dir", default="data", help="Local data directory")
    parser.add_argument("--execute", action="store_true", help="Actually upload (default: dry-run)")
    parser.add_argument("--bucket", default=None, help="Override R2_BUCKET_NAME")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        sys.exit(1)

    print(f"Scanning {data_dir.absolute()}...")
    files = collect_files(data_dir)

    total_bytes = sum(p.stat().st_size for p, _ in files)
    print(f"\n{len(files)} files, {total_bytes / 1e9:.2f} GB total")

    if not args.execute:
        print("\nDRY RUN — pass --execute to upload")
        print("\nFiles to upload:")
        for path, key in files:
            size = path.stat().st_size
            print(f"  {key}  ({size / 1e6:.1f} MB)")
        return

    # Init R2 client
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = args.bucket or os.environ.get("R2_BUCKET_NAME", "diffviews")

    if not all([account_id, access_key, secret_key]):
        print("Error: R2 credentials not set"
              " (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)")
        sys.exit(1)

    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    print(f"\nUploading to bucket '{bucket}'...")
    uploaded = 0
    failed = 0
    for i, (path, key) in enumerate(files, 1):
        size = path.stat().st_size
        print(f"  [{i}/{len(files)}] {key} ({size / 1e6:.1f} MB)...", end=" ", flush=True)
        try:
            client.upload_file(str(path), bucket, key)
            print("OK")
            uploaded += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone: {uploaded} uploaded, {failed} failed")


if __name__ == "__main__":
    main()

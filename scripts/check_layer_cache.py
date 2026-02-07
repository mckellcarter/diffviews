#!/usr/bin/env python3
"""Check which layer caches are seeded on R2.

Usage:
    python scripts/check_layer_cache.py
    python scripts/check_layer_cache.py --model dmd2

Requires env vars: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffviews.data.r2_cache import R2DataStore


def list_cached_layers(store: R2DataStore, model: str) -> list:
    """List layer names that have caches on R2."""
    prefix = f"data/{model}/layer_cache/"
    keys = store.list_objects(prefix)

    # Extract unique layer names from keys like "data/dmd2/layer_cache/encoder_bottleneck.csv"
    layers = set()
    for key in keys:
        filename = key.split("/")[-1]
        if filename.endswith(".csv"):
            layers.add(filename[:-4])

    return sorted(layers)


def main():
    parser = argparse.ArgumentParser(description="Check layer cache status on R2")
    parser.add_argument("--model", help="Check specific model (dmd2, edm)")
    args = parser.parse_args()

    store = R2DataStore()
    if not store.enabled:
        print("Error: R2 not configured")
        print("  Required: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(1)

    models = [args.model] if args.model else ["dmd2", "edm"]

    for model in models:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        layers = list_cached_layers(store, model)

        if layers:
            print(f"\nCached layers ({len(layers)}):")
            for layer in layers:
                print(f"  ✓ {layer}")
        else:
            print("\n  No cached layers found")


if __name__ == "__main__":
    main()

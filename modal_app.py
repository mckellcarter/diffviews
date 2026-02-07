"""
Modal serverless GPU entry point for diffviews.

Replaces HF Spaces ZeroGPU — single A10G container serving Gradio.
Data from Cloudflare R2, persisted on Modal Volume.

Usage:
    modal serve modal_app.py   # dev
    modal deploy modal_app.py  # prod
"""

import os
from pathlib import Path

import modal

app = modal.App("diffviews")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.0",
        "tqdm>=4.60.0",
        "numba==0.58.1",
        "gradio>=6.0.0",
        "plotly>=5.18.0",
        "matplotlib>=3.5.0",
        "huggingface_hub>=0.25.0",
        "boto3>=1.28.0",
        "scipy>=1.7.0",
        # cuML GPU acceleration (auto-detected by umap_backend.py)
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install("cuml-cu12>=25.02", "cupy-cuda12x>=12.0")
    # TODO: revert to @main before merging
    .pip_install("diffviews @ git+https://github.com/mckellcarter/diffviews.git@feature/modal-migrate")
)

vol = modal.Volume.from_name("diffviews-data", create_if_missing=True)

r2_secret = modal.Secret.from_name("R2_ACCESS")

# --- Data constants (same as app.py) ---

CHECKPOINT_URLS = {
    "dmd2": (
        "https://huggingface.co/mckell/diffviews-dmd2-checkpoint/"
        "resolve/main/dmd2-imagenet-64-10step.pkl"
    ),
    "edm": (
        "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
        "edm-imagenet-64x64-cond-adm.pkl"
    ),
}
CHECKPOINT_FILENAMES = {
    "dmd2": "dmd2-imagenet-64-10step.pkl",
    "edm": "edm-imagenet-64x64-cond-adm.pkl",
}
DATA_DIR = Path("/data")


# --- Data helpers ---

def get_pca_components() -> int | None:
    val = os.environ.get("DIFFVIEWS_PCA_COMPONENTS", "50")
    if val.lower() in ("0", "none", "off", ""):
        return None
    return int(val)


def download_data(output_dir: Path) -> None:
    """Download model data: R2 first, HF fallback."""
    from diffviews.data.r2_cache import R2DataStore

    store = R2DataStore()
    if store.enabled:
        print("Downloading data from R2...")
        for model in ["dmd2", "edm"]:
            store.download_model_data(model, output_dir)
        return

    from huggingface_hub import snapshot_download

    print("Downloading data from HF (R2 unavailable)...")
    snapshot_download(
        repo_id="mckell/diffviews_demo_data",
        repo_type="dataset",
        local_dir=output_dir,
        revision="main",
    )


def download_checkpoint(output_dir: Path, model: str) -> None:
    """Download checkpoint: R2 first, URL fallback."""
    if model not in CHECKPOINT_URLS:
        return

    ckpt_dir = output_dir / model / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filename = CHECKPOINT_FILENAMES[model]
    filepath = ckpt_dir / filename
    if filepath.exists():
        print(f"Checkpoint exists: {filepath}")
        return

    from diffviews.data.r2_cache import R2DataStore

    store = R2DataStore()
    r2_key = f"data/{model}/checkpoints/{filename}"
    if store.enabled and store.download_file(r2_key, filepath):
        print(f"Checkpoint from R2: {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")
        return

    import urllib.request

    url = CHECKPOINT_URLS[model]
    print(f"Downloading {model} checkpoint from URL...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  Done ({filepath.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  Error: {e}")


def regenerate_umap(data_dir: Path, model: str) -> bool:
    """Rebuild UMAP pkl from csv+activations for numba compatibility."""
    from diffviews.processing.umap import (
        load_dataset_activations,
        compute_umap,
        save_embeddings,
    )
    import json

    model_dir = data_dir / model
    activation_dir = model_dir / "activations" / "imagenet_real"
    metadata_path = model_dir / "metadata" / "imagenet_real" / "dataset_info.json"
    embeddings_dir = model_dir / "embeddings"

    if not activation_dir.exists() or not metadata_path.exists():
        print(f"  Skipping UMAP regen for {model}: missing activations")
        return False

    csv_files = list(embeddings_dir.glob("*.csv"))
    if not csv_files:
        print(f"  Skipping UMAP regen for {model}: no embeddings CSV")
        return False

    csv_path = csv_files[0]
    json_path = csv_path.with_suffix(".json")

    umap_params = {"n_neighbors": 15, "min_dist": 0.1, "layers": ["encoder_bottleneck", "midblock"]}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            umap_params = json.load(f)

    pca_components = get_pca_components()
    print(f"  Regenerating UMAP for {model} (pca={pca_components})...")

    try:
        activations, metadata_df = load_dataset_activations(activation_dir, metadata_path)
        print(f"    {activations.shape[0]} samples x {activations.shape[1]} dims")

        embeddings, reducer, scaler, pca_reducer = compute_umap(
            activations,
            n_neighbors=umap_params.get("n_neighbors", 15),
            min_dist=umap_params.get("min_dist", 0.1),
            normalize=True,
            pca_components=pca_components,
        )

        save_embeddings(
            embeddings, metadata_df, csv_path, umap_params, reducer, scaler, pca_reducer
        )
        print(f"    UMAP pkl regenerated: {csv_path.with_suffix('.pkl')}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def _umap_pkl_ok(pkl_path: Path) -> bool:
    """Check if UMAP pkl loads and transforms without numba errors."""
    try:
        import pickle
        import numpy as np
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        reducer = data.get("reducer")
        if reducer is None:
            return True
        # Derive correct input dims from scaler/pca/reducer
        scaler = data.get("scaler")
        pca = data.get("pca_reducer")
        if scaler and hasattr(scaler, "n_features_in_"):
            n_features = scaler.n_features_in_
        elif pca and hasattr(pca, "n_features_in_"):
            n_features = pca.n_features_in_
        else:
            n_features = reducer.n_features_in_ if hasattr(reducer, "n_features_in_") else 50
        # Dummy transform triggers numba JIT — fails if incompatible
        dummy = np.random.randn(1, n_features).astype(np.float32)
        if scaler:
            dummy = scaler.transform(dummy)
        if pca:
            dummy = pca.transform(dummy)
        reducer.transform(dummy)
        return True
    except Exception as e:
        print(f"  pkl check failed ({pkl_path.name}): {e}")
        return False


def ensure_data_ready(data_dir: Path) -> None:
    """Download data + checkpoints if missing, regenerate UMAP pkls."""
    models_ready = []
    for model in ["dmd2", "edm"]:
        config = data_dir / model / "config.json"
        emb_dir = data_dir / model / "embeddings"
        img_dir = data_dir / model / "images" / "imagenet_real"
        if (
            config.exists()
            and emb_dir.exists()
            and list(emb_dir.glob("*.csv"))
            and img_dir.exists()
            and list(img_dir.glob("sample_*.png"))
        ):
            models_ready.append(model)

    if not models_ready:
        print("Data not found, downloading...")
        download_data(data_dir)
    else:
        print(f"Data present: {models_ready}")

    for model in CHECKPOINT_URLS:
        download_checkpoint(data_dir, model)

    print("\nChecking UMAP pkls...")
    for model in ["dmd2", "edm"]:
        emb_dir = data_dir / model / "embeddings"
        if not emb_dir.exists() or not list(emb_dir.glob("*.csv")):
            continue
        pkl_files = list(emb_dir.glob("*.pkl"))
        if pkl_files and _umap_pkl_ok(pkl_files[0]):
            print(f"  {model}: pkl valid, skipping refit")
            continue
        regenerate_umap(data_dir, model)


def get_device() -> str:
    override = os.environ.get("DIFFVIEWS_DEVICE")
    if override:
        return override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- Modal entry ---

@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": vol},
    secrets=[r2_secret],
    timeout=600,
    scaledown_window=120,
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    """Called once per container to build the ASGI app."""
    import gradio as gr
    from fastapi import FastAPI
    from diffviews.visualization.app import (
        GradioVisualizer,
        create_gradio_app,
        CUSTOM_CSS,
        PLOTLY_HANDLER_JS,
    )

    data_dir = DATA_DIR
    device = get_device()

    print("=" * 50)
    print("DiffViews on Modal")
    print(f"  data: {data_dir}  device: {device}")
    print("=" * 50)

    ensure_data_ready(data_dir)
    vol.commit()

    print("\nInitializing visualizer...")
    visualizer = GradioVisualizer(data_dir=data_dir, device=device)

    print("Creating Gradio app...")
    demo = create_gradio_app(visualizer)
    demo.queue(max_size=20)

    print("Setup complete — ready to serve.")
    return gr.mount_gradio_app(
        app=FastAPI(),
        blocks=demo,
        path="/",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        js=PLOTLY_HANDLER_JS,
    )

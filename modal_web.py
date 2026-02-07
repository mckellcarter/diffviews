"""
Modal CPU web server for diffviews — serves Gradio UI.

Handles cached layer visualization locally.
Calls modal_gpu.py remotely for generation/extraction.

Usage:
    modal serve modal_web.py   # dev (also starts GPU worker)
    modal deploy modal_web.py  # prod
"""

import os
from pathlib import Path

import modal

app = modal.App("diffviews-web")

# CPU-only image (no torch/cuML needed for UI)
cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "gradio>=6.0.0",
        "plotly>=5.18.0",
        "matplotlib>=3.5.0",
        "boto3>=1.28.0",
    )
    # TODO: revert to @main before merging
    .pip_install("diffviews @ git+https://github.com/mckellcarter/diffviews.git@feature/modal-migrate")
)

vol = modal.Volume.from_name("diffviews-data", create_if_missing=True)
r2_secret = modal.Secret.from_name("R2_ACCESS")

DATA_DIR = Path("/data")


def download_data(output_dir: Path) -> None:
    """Download data from R2 (CPU-only, no checkpoints needed)."""
    from diffviews.data.r2_cache import R2DataStore

    store = R2DataStore()
    if not store.enabled:
        print("Warning: R2 not configured")
        return

    for model in ["dmd2", "edm"]:
        config = output_dir / model / "config.json"
        if not config.exists():
            print(f"Downloading {model} data from R2...")
            store.download_model_data(model, output_dir)


@app.function(
    image=cpu_image,
    volumes={"/data": vol},
    secrets=[r2_secret],
    timeout=600,
    scaledown_window=300,  # Keep CPU container warm longer
    max_containers=2,  # Allow multiple CPU containers
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    """Build the ASGI app (CPU container)."""
    import gradio as gr
    from fastapi import FastAPI

    from diffviews.visualization.app import (
        GradioVisualizer,
        create_gradio_app,
        CUSTOM_CSS,
        PLOTLY_HANDLER_JS,
    )
    from diffviews.visualization import gpu_ops

    # Configure GPU ops to use remote worker (lookup from deployed app)
    try:
        GPUWorker = modal.Cls.from_name("diffviews-gpu", "GPUWorker")
        gpu_worker = GPUWorker()
        gpu_ops.set_remote_gpu_worker(gpu_worker)
        print("Remote GPU worker connected.")
    except Exception as e:
        print(f"Warning: Could not connect to GPU worker: {e}")
        print("Running in local mode (GPU ops will fail without GPU).")

    print("=" * 50)
    print("DiffViews Web (CPU) on Modal")
    print("=" * 50)

    # Download cached data (no checkpoints on CPU)
    download_data(DATA_DIR)
    vol.commit()

    # Initialize visualizer in CPU-only mode
    print("\nInitializing visualizer (CPU mode)...")
    visualizer = GradioVisualizer(
        data_dir=DATA_DIR,
        device="cpu",  # No GPU on this container
    )

    print("Creating Gradio app...")
    demo = create_gradio_app(visualizer)
    demo.queue(max_size=20)

    print("Setup complete — ready to serve.")
    print("GPU operations will be routed to remote worker.")

    return gr.mount_gradio_app(
        app=FastAPI(),
        blocks=demo,
        path="/",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        js=PLOTLY_HANDLER_JS,
    )

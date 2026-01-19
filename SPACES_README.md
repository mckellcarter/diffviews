---
title: DiffViews
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
python_version: "3.10"
pinned: false
license: mit
---

# DiffViews - Diffusion Activation Visualizer

Interactive visualization of diffusion model activations projected to 2D via UMAP.

## Features
- Explore activation space of diffusion models
- Select points and find nearest neighbors
- Generate images from averaged neighbor activations
- Visualize denoising trajectories

## Usage
1. Hover over points to preview samples
2. Click to select a point
3. Click nearby points or use "Suggest KNN" to add neighbors
4. Click "Generate from Neighbors" to create new images

## Note
First launch downloads ~2.5GB of data and checkpoints. Generation on CPU takes ~30-60s per image.

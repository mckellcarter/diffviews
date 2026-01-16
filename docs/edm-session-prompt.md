# Session Prompt: Implement EDM Model Support

## Context Setup

Please read these files to understand the current state:

1. **Implementation Plan**: `docs/edm-support-plan.md` - Full plan for adding EDM support
2. **Current Visualizer**: `diffviews/visualization/app.py` - Working version (reverted after failed EDM attempt)
3. **Model Configs**: `data/dmd2/config.json` and `data/edm/config.json`
4. **Tests**: `tests/test_visualizer.py` - Tests for model switching functionality
5. **Reference Metadata**: Check `/Users/mckell/Documents/GitHub/DMD2/visualizer/data/edm-imagenet-64-sigmas50_14_1.0_0.05/umap_raw/` for how JSON carries adapter/checkpoint info

## Task

Implement multi-model support (DMD2 + EDM) for the diffviews visualization app.

## Critical Constraint

The previous attempt broke the app by adding outputs to `update_plot` callback. The solution uses **separate callbacks**:
- `update_plot` stays simple (2 outputs only)
- New `handle_model_switch` callback triggers model switching
- New `reset_selection_on_model_switch` callback resets stores

## Steps

1. First, run the current app to verify it works: `python -m diffviews.visualization.app --data_dir=demo_data --embeddings=demo_data/embeddings/demo_embeddings.csv`

2. Follow the implementation phases in `docs/edm-support-plan.md`

3. Test frequently - run the app after each major change to catch regressions early

## Key Questions to Resolve

- Where are checkpoints stored? Need paths for both DMD2 and EDM
- Should config.json include checkpoint path, or pass via CLI?
- How does the reference DMD2 repo propagate adapter/checkpoint through the pipeline?

# EDM Model Support Implementation Plan

## Background

The visualization app broke when adding EDM support. Root cause: adding too many outputs to `update_plot` callback created conflicts with other callbacks writing to the same stores.

## Key Files

- `diffviews/visualization/app.py` - Main visualizer (reverted to working state)
- `data/dmd2/config.json` - DMD2 model config
- `data/edm/config.json` - EDM model config
- `tests/test_visualizer.py` - Tests for model switching
- Reference: `/Users/mckell/Documents/GitHub/DMD2/visualizer/data/edm-imagenet-64-sigmas50_14_1.0_0.05/umap_raw/*.json` - Shows how metadata carries adapter/checkpoint through pipeline

## Solution: Separate Callback Chain

### 1. Enhanced config.json Format
```json
{
  "adapter": "dmd2-imagenet-64",
  "checkpoint": "checkpoints/dmd2-imagenet-10step",
  "default_steps": 1,
  "sigma_max": 80.0,
  "sigma_min": 0.002
}
```

### 2. Keep `update_plot` Simple
- Only 2 outputs: figure, status
- Add Input from `model-switch-trigger` to refresh after switch
- Do NOT output to selection stores

### 3. New Dedicated Callbacks

**handle_model_switch:**
```python
@self.app.callback(
    Output("model-switch-trigger", "data"),
    Input("model-selector", "value"),
    State("current-model-store", "data"),
    prevent_initial_call=True
)
def handle_model_switch(new_model, current_model):
    if new_model != current_model:
        self.switch_model(new_model)
        return {"switched": True, "model": new_model}
    return dash.no_update
```

**reset_selection_on_model_switch:**
```python
@self.app.callback(
    Output("selected-point-store", "data", allow_duplicate=True),
    Output("manual-neighbors-store", "data", allow_duplicate=True),
    Output("neighbor-indices-store", "data", allow_duplicate=True),
    Output("current-model-store", "data", allow_duplicate=True),
    Input("model-switch-trigger", "data"),
    prevent_initial_call=True
)
def reset_selection_on_model_switch(trigger):
    if trigger and trigger.get("switched"):
        return None, [], None, trigger["model"]
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update
```

### 4. Model Discovery
- Scan data directory for subdirs with `config.json` + `embeddings/`
- Build `model_configs` dict with data_dir, adapter, checkpoint, defaults
- Auto-populate model selector dropdown

### 5. Generation Per-Model Config
- Get checkpoint_path from `model_configs[current_model]`
- Use model-specific sigma defaults
- Adapter loaded dynamically

## Implementation Phases

1. **Data Structure** - Extend config.json, update embeddings JSON
2. **Model Discovery** - Add discover_models(), modify __init__
3. **Callbacks** - Add new stores, handle_model_switch, reset callbacks
4. **UI** - Model selector dropdown, status updates
5. **Generation** - Per-model checkpoint/params
6. **Testing** - Run test_visualizer.py, manual testing

## CLI Changes
- Remove `--edm_data_dir`, `--edm_embeddings_path`
- Keep `--data_dir` as parent containing model subdirs
- Add `--model` for initial model selection

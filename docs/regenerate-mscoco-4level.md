# Regenerate MSCOCO Data with 4 Native Timestep Levels

Current MSCOCO data uses Karras sigma values (80, 14, 1.9, 0.6) which are converted at display time.
This doc describes regenerating with native DDPM timesteps.

## Target Levels

4 levels spanning the denoising process (1000 DDPM steps):

| Level | Noise % | DDPM Timestep |
|-------|---------|---------------|
| 1     | 100%    | 999           |
| 2     | 66%     | 666           |
| 3     | 33%     | 333           |
| 4     | 0.5%    | 5             |

## Steps

### 1. Update extraction script

In extraction config, replace sigma levels with native timesteps:

```python
# OLD (Karras sigma)
sigma_levels = [80.0, 14.0, 1.9, 0.6]

# NEW (DDPM timesteps)
timestep_levels = [999, 666, 333, 5]
```

### 2. Modify extractor to use native timesteps

The extractor needs to:
- Accept `timestep_levels` instead of `sigma_levels`
- Pass native timesteps directly to the adapter's forward pass
- Store `timestep` column (not `sigma`) in output parquet

```python
# In extraction loop
for t in timestep_levels:
    # MSCOCO adapter expects native timestep, not sigma
    activations = extract_at_timestep(adapter, samples, t)
    df["timestep"] = t  # or keep as "sigma" for backwards compat
```

### 3. Run extraction

```bash
python -m diffviews.scripts.extract_activations \
    --model mscoco \
    --timesteps 999,666,333,5 \
    --output data/mscoco/activations/
```

### 4. Regenerate aligned UMAP

```bash
python -m diffviews.scripts.compute_aligned_umap \
    --model mscoco \
    --layers mid_block \
    --output data/mscoco/embeddings/
```

### 5. Upload to R2

```bash
python -m diffviews.scripts.upload_to_r2 --model mscoco
```

### 6. Remove temp fix

After regenerating, remove the `_sigma_to_ddpm_timestep` conversion in:
- `diffviews/visualization/visualizer.py` (lines ~36-50, ~1643-1662)

## Verification

After regeneration, the 3D view legend should show:
- t=999
- t=666
- t=333
- t=5

(not sigma values like 80.29, 14.00, etc.)

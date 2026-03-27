# Future Considerations

## Config File Distribution

**Issue:** Model config files (`data/<model>/config.json`) are in gitignored `data/` directory. New users loading diffviews fresh must manually create these or download from R2.

**Current state:**
- Configs define: adapter name, checkpoint path, conditioning_type, default_guidance, sigma ranges
- R2 data download includes configs, so Modal/HF deployments work
- Local dev requires manual setup or running download scripts

**Potential solutions:**
1. Bundle default configs in package (e.g., `diffviews/configs/`) and copy on first run
2. Have adapters provide all defaults via `get_default_config()` - no config.json needed
3. Script to generate configs from adapter registry
4. Document required config format in README

## Auto-encode/decode for Latent Models

**Idea:** Automatically trigger VAE encode/decode based on `adapter.uses_latent` property instead of manual decode calls.

**Current state:** Generator explicitly calls `adapter.decode()` which is identity for pixel-space models.

**Consideration:** Could make this more implicit but current explicit approach is clear and works.

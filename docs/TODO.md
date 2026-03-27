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

## Text Encoding Should Live in Adapter

**Issue:** Text encoding for T2I models is currently handled in both diffviews and adapt_diff. Should be fully encapsulated in adapter.

**Current state:**
- Adapter has `prepare_conditioning(text=...)` which loads and runs CLIP encoder
- diffviews passes caption string through to adapter (correct)
- SD2 text encoder requires HF authentication (gated model)
- OpenCLIP ViT-H-14 doesn't produce text-reflective images (different weights?)

**Goal:** All text encoding lives in adapter, diffviews just passes caption strings. Adapter handles encoder loading, tokenization, and any model-specific quirks.

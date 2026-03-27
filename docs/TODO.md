# Future Considerations

## Config File Distribution

**Issue:** Model config files (`data/<model>/config.json`) are in gitignored `data/` directory. New users loading diffviews fresh must manually create these or download from R2.

**Current state:**
- Configs define: adapter name, checkpoint path, conditioning_type, default_guidance, sigma ranges
- R2 data download includes configs, so Modal/HF deployments work
- Local dev requires manual setup or running download scripts

**Recommended solution:** Have adapters provide all defaults via `get_default_config()`.

- Adapter already knows its own defaults (guidance, sigmas, conditioning_type, etc.)
- No config.json needed - diffviews queries adapter directly
- Config files become optional overrides only
- Avoids committing configs without data, merge conflicts, and config/data coupling

## Auto-encode/decode for Latent Models

**Idea:** Automatically trigger VAE encode/decode based on `adapter.uses_latent` property instead of manual decode calls.

**Current state:** Generator explicitly calls `adapter.decode()` which is identity for pixel-space models.

**Consideration:** Could make this more implicit but current explicit approach is clear and works.

## Text Encoding Should Live in Adapter

**Status:** ✅ Resolved

**Solution:**
- Adapter's `prepare_conditioning(text=...)` handles all text encoding
- Uses `sd2-community/stable-diffusion-2-1` text encoder (community fork after SD2 deprecation)
- diffviews passes caption string only, no CLIP/encoding code
- Removed unused `encode_text()`, `get_uncond_text_embedding()` from visualizer
- Removed `text_encoder`, `text_tokenizer` fields from ModelData

# Continuation Prompt: Modal Transition (M1→M3)

Paste this after clearing context to resume work.

---

## Prompt

I'm migrating the diffviews visualizer from HF Spaces to Modal with Cloudflare R2 for data hosting. The full plan is in `docs/modal-transition-plan.md` — read it first.

### What's Done

**M1 (R2 layer cache):** Complete, merged to main (PR #66). PCA pre-reduction, R2LayerCache for layer embeddings, full UMAP refit path, E2E verified on HF.

**M2 (CF data hosting):** Complete, E2E verified on HF. R2DataStore in `diffviews/data/r2_cache.py`, R2-first downloads in `app.py`, CLI `--source` flag, seeding script. Bug fixes: concurrent downloads, path fix, OOM, trajectory projection.

**M3 (Modal compute):** Complete, E2E verified on Modal. All features working.

**`modal_app.py`** — Modal entry point:
- Image: debian_slim + pip_install deps + `diffviews@git+...@main`
- `@app.function(gpu="A10G", max_containers=1)` → `@modal.concurrent(max_inputs=100)` → `@modal.asgi_app()`
- `ensure_data_ready()` downloads from R2 (HF fallback), checks/refits UMAP pkls
- `_umap_pkl_ok()` reads `n_features_in_` from scaler for correct dummy dims, tests numba JIT compat
- Gradio 6: `mount_gradio_app(theme=, css=, js=)` — NOT manual `_set_html_css_theme_variables()`
- Volume `diffviews-data` at `/data`, Secret `R2_ACCESS`

**`diffviews/visualization/app.py`** — ZeroGPU comments cleaned (4 locations), functions unchanged.

**`requirements.txt`** — removed `spaces`, added `modal>=0.73.0`.

**Tests:** 148 passing (1 pre-existing flaky `test_cross_sample_masking`). Lint 9.81/10.

### Key Architecture Notes

- `mount_gradio_app(theme=, css=, js=)` is required for Gradio 6 ASGI. Manual Blocks attr assignment + `_set_html_css_theme_variables()` leaves `body_css=None` → Jinja2 crash.
- `_umap_pkl_ok()` must use `scaler.n_features_in_` for dummy shape — hardcoded 50 causes shape mismatch at scaler, always fails.
- `max_containers=1` required — Gradio SSE needs sticky sessions.
- `@modal.concurrent` IS compatible with `@modal.asgi_app()` (canonical Modal pattern).
- `app.py` kept for HF Spaces dual deployment.

### What's Next

1. PR `feature/modal-transition` → main
2. `modal deploy modal_app.py` for production
3. Future: cuML GPU UMAP, auth, `modal.web_server` alternative if ASGI issues arise

### Branch

`feature/modal-transition` (from `main`)

Please read the plan file and continue from where we left off.

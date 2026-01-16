# Continuation Prompt for Gradio Port

---

I'm continuing a Gradio port of a Dash visualization app.

Repo: diffviews
Branch: feature/gradio-port-phase2-selection

**URGENT: Switch from ScatterPlot to Plotly**

ScatterPlot (Altair) has critical issues:
- Can't control plot size
- Click returns coordinates not row indices
- Zoom triggers broken events

Plan: Use `gr.Plot` with Plotly + JS bridge for clicks.
See detailed plan below.

Start by reading @diffviews/visualization/gradio_app.py

---

## Plotly Migration Plan

**JS Bridge Pattern:**
1. `gr.Plot` displays Plotly figure
2. Hidden `gr.Textbox` receives click data from JS
3. JS attaches `plotly_click` handler, writes `{pointIndex, x, y}` to textbox
4. Textbox `.change()` triggers Python callback

**Key steps:**
1. Add `create_umap_figure()` method (replace `get_plot_dataframe()`)
2. Add hidden click_data_box + CLICK_HANDLER_JS
3. Replace `gr.ScatterPlot` with `gr.Plot`
4. Wire `click_data_box.change()` instead of `.select()`
5. Update all callbacks that return plot updates

**Reference:** Dash app at `diffviews/visualization/app.py` lines 802-816 (figure), 962-1005 (click handling)

---

## Current State

- `gradio_app.py` has turbo colormap working
- KNN suggest + distance display working
- Click selection BROKEN (ScatterPlot returns coords not indices)
- 87 tests pass

```bash
python -m pytest tests/test_gradio_visualizer.py -v
python -m diffviews.visualization.gradio_app --data-dir data
```

# Continuation Prompt for Gradio Port

---

I'm continuing a Gradio port of a Dash visualization app.

Repo: diffviews
Branch: feature/gradio-port-phase2-selection

**Phase 2 COMPLETE: Plotly + JS Bridge for Click Handling**

Start by reading @diffviews/visualization/gradio_app.py

---

## What's Working

- Plotly scatter plot via `gr.Plot` with `go.Scattergl`
- Click handling via JS bridge pattern (hidden textbox)
- Point selection (red ring highlight)
- Manual neighbor toggling (cyan ring)
- KNN suggestions with distance display (lime ring)
- Neighbor gallery with class labels and distances
- Class filtering, model switching, clear buttons

## JS Bridge Pattern (for future reference)

**Key learnings:**
1. Inject JS via `head=` param in `gr.Blocks()` (not `js=` on events, not `gr.HTML`)
2. Use `.input()` event on textbox (not `.change()`)
3. Include textbox in inputs list explicitly - Gradio doesn't auto-pass
4. Use `go.Scattergl` with `.tolist()` for Gradio compatibility
5. Benign "too many arguments" warning can be ignored

**Pattern:**
```python
# In gr.Blocks:
head=f"<script>{CLICK_HANDLER_JS}</script>"

# Hidden textbox:
click_data_box = gr.Textbox(value="", elem_id="click-data-box", visible=False)

# JS attaches plotly_click handler, writes JSON to textbox:
CLICK_HANDLER_JS = """
plotDiv.on('plotly_click', function(data) {
    clickBox.value = JSON.stringify({pointIndex: point.customdata, ...});
    clickBox.dispatchEvent(new Event('input', { bubbles: true }));
});
"""

# Python handler wired to .input():
click_data_box.input(handler, inputs=[click_data_box, ...], outputs=[...])
```

---

## Current State

- `gradio_app.py` (~1100 lines) with Plotly working
- 23 Gradio-specific tests pass
- Next: Phase 3 (generation) or Phase 4 (polish)

**Key files:**
- `diffviews/visualization/gradio_app.py` - main app
- `diffviews/visualization/app.py` - Dash reference
- `tests/test_gradio_visualizer.py` - tests

```bash
python -m pytest tests/test_gradio_visualizer.py -v
python -m diffviews.visualization.gradio_app --data-dir data
```

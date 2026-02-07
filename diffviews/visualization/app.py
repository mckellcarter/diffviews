"""
Gradio-based diffusion activation visualizer.
Port of the Dash visualization app with multi-user support.
"""

import argparse
import json

import gradio as gr

from diffviews.utils.device import get_device

# Re-exports for backward compatibility
from .models import ModelData
from .layout import CUSTOM_CSS, PLOTLY_HANDLER_JS
from .gpu_ops import (
    _app_visualizer,
    set_visualizer,
    get_visualizer,
    _generate_on_gpu,
    _extract_layer_on_gpu,
    set_remote_gpu_worker,
    is_hybrid_mode,
)
from .visualizer import GradioVisualizer


def create_gradio_app(visualizer: GradioVisualizer) -> gr.Blocks:
    """Create Gradio Blocks app.

    Note: In Gradio 6, theme/css/head are passed to launch() not Blocks().
    Use CUSTOM_CSS and PLOTLY_HANDLER_JS module constants when calling launch().
    """
    set_visualizer(visualizer)
    with gr.Blocks(
        title="Diffusion Activation Visualizer",
    ) as app:

        # Per-session state
        current_model = gr.State(value=visualizer.default_model)  # Model selection per session
        selected_idx = gr.State(value=None)
        manual_neighbors = gr.State(value=[])
        knn_neighbors = gr.State(value=[])
        knn_distances = gr.State(value={})  # {idx: distance} for KNN neighbors
        highlighted_class = gr.State(value=None)
        trajectory_coords = gr.State(value=[])  # [[(x, y, sigma), ...], ...] list of trajectories
        intermediate_images = gr.State(value=[])  # [[(img, sigma), ...]] list (trajs) or list (steps) for hover
        animation_frame = gr.State(value=-1)  # Current animation frame (-1 = showing final)
        generation_infos = gr.State(value=[])  # [{class_id, class_name, n_steps, final_image}, ...] per traj

        # Get initial model data for default values
        default_model_data = visualizer.get_model(visualizer.default_model)
        initial_sample_count = len(default_model_data.df) if default_model_data else 0

        gr.Markdown("# Diffusion Activation Visualizer")

        with gr.Row(elem_id="main-row"):
            # Left column (sidebar)
            with gr.Column(scale=1, elem_id="left-sidebar"):
                # Model selector + status
                with gr.Row(elem_id="model-row"):
                    gr.Markdown("**Model**", elem_id="model-label")
                    if len(visualizer.model_configs) > 1:
                        model_dropdown = gr.Dropdown(
                            choices=list(visualizer.model_configs.keys()),
                            value=visualizer.default_model,
                            show_label=False,
                            interactive=True,
                            elem_id="model-dropdown",
                        )
                    else:
                        model_dropdown = gr.Dropdown(
                            choices=[visualizer.default_model] if visualizer.default_model else [],
                            value=visualizer.default_model,
                            show_label=False,
                            visible=len(visualizer.model_configs) > 0,
                            elem_id="model-dropdown",
                        )
                with gr.Row(elem_id="layer-row"):
                    gr.Markdown("**Layer**", elem_id="layer-label")
                    layer_dropdown = gr.Dropdown(
                        choices=visualizer.get_layer_choices(visualizer.default_model) if visualizer.default_model else [],
                        value=visualizer.get_default_layer_label(visualizer.default_model) if visualizer.default_model else None,
                        show_label=False,
                        interactive=True,
                        elem_id="layer-dropdown",
                    )
                _default_layer = visualizer.get_default_layer_label(visualizer.default_model) if visualizer.default_model else None
                status_text = gr.Markdown(
                    f"Showing {initial_sample_count} samples"
                    + (f" ({visualizer.default_model})" if visualizer.default_model else "")
                    + (f" — layer: {_default_layer}" if _default_layer else ""),
                    elem_id="status-text"
                )
                model_status = gr.Markdown("", visible=False)

                # Preview section (updated on hover)
                with gr.Group():
                    gr.Markdown("### Preview")
                    preview_image = gr.Image(
                        label=None, show_label=False, elem_id="preview-image"
                    )
                    preview_details = gr.Markdown(
                        "Hover over a point to preview", elem_id="preview-details"
                    )

                # Class filter
                with gr.Group():
                    gr.Markdown("### Class Filter")
                    class_dropdown = gr.Dropdown(
                        choices=visualizer.get_class_options(visualizer.default_model) if visualizer.default_model else [],
                        label="Select class",
                        interactive=True,
                    )
                    clear_class_btn = gr.Button("Clear Highlight", size="sm")
                    class_status = gr.Markdown("")

                # Selected sample (moved from right sidebar)
                with gr.Group():
                    gr.Markdown("### Selected Sample")
                    selected_image = gr.Image(
                        label=None, show_label=False, height=150, elem_id="selected-image"
                    )
                    selected_details = gr.Markdown(
                        "Click a point to select", elem_id="selected-details"
                    )
                    clear_selection_btn = gr.Button("Clear Selection", size="sm")

            # Center column (main plot)
            with gr.Column(scale=3, min_width=500, elem_id="center-column"):
                # Use Plotly via gr.Plot for proper click handling
                umap_plot = gr.Plot(
                    value=visualizer.create_umap_figure(visualizer.default_model) if visualizer.default_model else None,
                    elem_id="umap-plot",
                    show_label=False,
                )
                # Hidden textboxes for JS bridge (after plot to not affect top alignment)
                # Note: visible=True but hidden via CSS - Gradio 6 doesn't render visible=False
                click_data_box = gr.Textbox(
                    value="",
                    elem_id="click-data-box",
                    visible=True,  # Hidden via CSS, must be in DOM for JS bridge
                )
                hover_data_box = gr.Textbox(
                    value="",
                    elem_id="hover-data-box",
                    visible=True,  # Hidden via CSS, must be in DOM for JS bridge
                )

            # Right column (generation & neighbors)
            with gr.Column(scale=1, elem_id="right-sidebar"):
                # Generation settings
                with gr.Group(elem_id="gen-group"):
                    gr.Markdown("### Generation")
                    # Parameters and buttons first (use model defaults)
                    default_steps = default_model_data.default_steps if default_model_data else 5
                    default_sigma_max = default_model_data.sigma_max if default_model_data else 80.0
                    default_sigma_min = default_model_data.sigma_min if default_model_data else 0.5
                    with gr.Row(elem_id="gen-params-row"):
                        num_steps_slider = gr.Number(
                            value=default_steps, label="Steps",
                            elem_id="num-steps", min_width=50, precision=0
                        )
                        mask_steps_slider = gr.Number(
                            value=visualizer.mask_steps, label="Mask",
                            elem_id="mask-steps", min_width=50, precision=0
                        )
                        guidance_slider = gr.Number(
                            value=visualizer.guidance_scale, label="CFG",
                            elem_id="guidance", min_width=50
                        )
                        sigma_max_input = gr.Number(
                            value=default_sigma_max, label="σ max",
                            elem_id="sigma-max", min_width=50
                        )
                        sigma_min_input = gr.Number(
                            value=default_sigma_min, label="σ min",
                            elem_id="sigma-min", min_width=50
                        )
                    noise_mode_dropdown = gr.Dropdown(
                        choices=["stochastic noise", "fixed noise", "zero noise"],
                        value="stochastic noise",
                        show_label=False,
                        elem_id="noise-mode",
                    )
                    generate_btn = gr.Button("Generate from Neighbors", variant="primary")
                    gen_traj_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        show_label=False,
                        elem_id="gen-traj-dropdown",
                        interactive=True,
                    )
                    # Generated output
                    generated_image = gr.Image(
                        label=None, show_label=False, elem_id="generated-image"
                    )
                    # Denoising steps gallery with frame nav
                    intermediate_gallery = gr.Gallery(
                        label="Denoising Steps",
                        show_label=True,
                        columns=5,
                        rows=1,
                        height=70,
                        object_fit="contain",
                        buttons=[],  # Hide download/share buttons
                        allow_preview=False,
                        elem_id="intermediate-gallery",
                    )
                    with gr.Row():
                        prev_frame_btn = gr.Button("◀", size="sm", min_width=40)
                        next_frame_btn = gr.Button("▶", size="sm", min_width=40)
                    clear_gen_btn = gr.Button("Clear", size="sm")

                # Neighbor list
                with gr.Group():
                    gr.Markdown("### Neighbors")
                    with gr.Row(elem_id="knn-row"):
                        gr.Markdown("**K-neighbors**", elem_id="knn-label")
                        knn_k_slider = gr.Number(
                            value=5, show_label=False, precision=0, minimum=1,
                            elem_id="knn-input", min_width=50
                        )
                        suggest_btn = gr.Button("Suggest KNN", size="sm")
                    neighbor_gallery = gr.Gallery(
                        label=None,
                        show_label=False,
                        columns=2,
                        height="auto",
                        object_fit="contain",
                        allow_preview=True,
                        elem_id="neighbor-gallery",
                    )
                    neighbor_info = gr.Markdown("No neighbors selected")
                    clear_neighbors_btn = gr.Button("Clear Neighbors", size="sm")

        # --- Event Handlers ---

        def on_load():
            """No-op - KNN models are already fit during init."""
            pass

        def build_neighbor_gallery(model_name, sel_idx, man_n, knn_n, knn_dist):
            """Build neighbor gallery and info text."""
            model_data = visualizer.get_model(model_name)
            if sel_idx is None or model_data is None:
                return [], "No neighbors selected"

            man_n = man_n or []
            knn_n = knn_n or []
            knn_dist = knn_dist or {}

            # Combine neighbors: KNN first (sorted by distance), then manual
            all_neighbors = []
            knn_with_dist = [(idx, knn_dist.get(idx, 999)) for idx in knn_n if idx not in man_n]
            knn_with_dist.sort(key=lambda x: x[1])
            all_neighbors.extend([idx for idx, _ in knn_with_dist])
            all_neighbors.extend(man_n)

            if not all_neighbors:
                return [], "Click points or use Suggest"

            images = []
            for idx in all_neighbors[:20]:
                if idx < len(model_data.df):
                    sample = model_data.df.iloc[idx]
                    img = visualizer.get_image(model_name, sample["image_path"])
                    if img is not None:
                        if "class_label" in sample:
                            cls_id = int(sample["class_label"])
                            cls_name = visualizer.get_class_name(cls_id)
                            label = f"{cls_id}: {cls_name}"
                        else:
                            label = f"#{idx}"
                        if idx in knn_dist:
                            label += f" (d={knn_dist[idx]:.2f})"
                        elif idx in man_n:
                            label += " (manual)"
                        images.append((img, label))

            knn_count = len([n for n in knn_n if n not in man_n])
            man_count = len(man_n)
            info = f"{len(all_neighbors)} neighbors"
            if knn_count > 0:
                info += f" ({knn_count} KNN"
            if man_count > 0:
                info += f", {man_count} manual" if knn_count > 0 else f" ({man_count} manual"
            if knn_count > 0 or man_count > 0:
                info += ")"
            return images, info

        def on_hover_data(hover_json, intermediates, model_name):
            """Handle plot hover via JS bridge - update preview panel.

            Handles two types:
            - Sample hover: show sample image from dataset
            - Trajectory hover: show intermediate image from generation
            """
            if not hover_json:
                return gr.update(), gr.update()

            try:
                hover_data = json.loads(hover_json)
            except (json.JSONDecodeError, TypeError, ValueError):
                return gr.update(), gr.update()

            hover_type = hover_data.get("type", "sample")

            # Trajectory point hover - show intermediate image
            if hover_type == "trajectory":
                step_idx = hover_data.get("stepIdx")
                traj_idx = hover_data.get("trajIdx", 0)
                sigma = hover_data.get("sigma", "?")

                if traj_idx and step_idx is not None: 
                    step_idx = int(step_idx)
                    traj_idx = int(traj_idx)

                if intermediates is not None:
                    step_idx = int(step_idx)
                    if 0 <= step_idx-1 < len(intermediates[traj_idx]):
                        img, stored_sigma = intermediates[traj_idx][step_idx-1]
                        details = f"**Trajectory {traj_idx + 1}, Step {step_idx}**\n\n"
                        details += f"σ = {stored_sigma:.1f}\n\n"
                        details += f"Coords: ({hover_data.get('x', 0):.2f}, {hover_data.get('y', 0):.2f})"
                        return img, details

                # No intermediates stored
                details = f"**Trajectory step {step_idx}**\n\nσ = {sigma}"
                return gr.update(), details

            # Sample hover - show dataset image
            model_data = visualizer.get_model(model_name)
            if model_data is None:
                return gr.update(), gr.update()

            point_idx = hover_data.get("pointIndex")
            if point_idx is None:
                return gr.update(), gr.update()
            point_idx = int(point_idx)

            if point_idx < 0 or point_idx >= len(model_data.df):
                return gr.update(), gr.update()

            sample = model_data.df.iloc[point_idx]
            img = visualizer.get_image(model_name, sample["image_path"])

            # Format details
            if "class_label" in sample:
                class_name = visualizer.get_class_name(int(sample["class_label"]))
            else:
                class_name = "N/A"
            details = f"**{sample['sample_id']}**<br>"
            if "class_label" in sample:
                details += f"Class: {int(sample['class_label'])}: {class_name}<br>"
            if "conditioning_sigma" in sample:
                details += f"σ = {sample['conditioning_sigma']:.1f}<br>"
            details += f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"

            return img, details

        def on_click_data(click_json, sel_idx, man_n, knn_n, knn_dist, high_class, traj, model_name):
            """Handle plot click via JS bridge - select point or toggle neighbor."""
            if not click_json:
                return (gr.update(),) * 9

            model_data = visualizer.get_model(model_name)
            if model_data is None:
                return (gr.update(),) * 9

            try:
                click_data = json.loads(click_json)
                # Only handle clicks on main samples trace (curve 0)
                # Ignore trajectory and other overlay traces
                if click_data.get("curveNumber", 0) != 0:
                    return (gr.update(),) * 9
                point_idx = click_data.get("pointIndex")
                if point_idx is None:
                    return (gr.update(),) * 9
                point_idx = int(point_idx)
            except (json.JSONDecodeError, TypeError, ValueError):
                return (gr.update(),) * 9

            if point_idx < 0 or point_idx >= len(model_data.df):
                return (gr.update(),) * 9

            knn_dist = knn_dist or {}

            # First click: select this point
            if sel_idx is None:
                sample = model_data.df.iloc[point_idx]
                img = visualizer.get_image(model_name, sample["image_path"])

                # Format details
                if "class_label" in sample:
                    class_name = visualizer.get_class_name(int(sample["class_label"]))
                else:
                    class_name = "N/A"
                details = f"**{sample['sample_id']}**<br>"
                if "class_label" in sample:
                    details += f"Class: {int(sample['class_label'])}: {class_name}<br>"
                if "conditioning_sigma" in sample:
                    details += f"σ = {sample['conditioning_sigma']:.1f}<br>"
                details += f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"

                # Build updated Plotly figure with selection (preserve trajectory)
                fig = visualizer.create_umap_figure(
                    model_name,
                    selected_idx=point_idx,
                    highlighted_class=high_class,
                    trajectory=traj if traj else None,
                )

                return (
                    img,           # selected_image
                    details,       # selected_details
                    point_idx,     # selected_idx
                    [],            # manual_neighbors
                    [],            # knn_neighbors
                    fig,           # umap_plot
                    [],            # neighbor_gallery
                    "Click points or use Suggest",  # neighbor_info
                    traj,          # trajectory_coords (preserved)
                )

            # Clicking same point: do nothing
            if point_idx == sel_idx:
                return (gr.update(),) * 9

            # Toggle neighbor (preserve trajectory)
            man_n = list(man_n) if man_n else []
            knn_n = list(knn_n) if knn_n else []
            at_limit = False

            if point_idx in man_n:
                man_n.remove(point_idx)
            elif point_idx in knn_n:
                knn_n.remove(point_idx)
            else:
                # Check total neighbor limit
                total = len(man_n) + len(knn_n)
                if total >= 20:
                    at_limit = True
                else:
                    man_n.append(point_idx)

            # Rebuild Plotly figure with updated highlights (preserve trajectory)
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )

            # Build gallery for neighbors
            gallery, info = build_neighbor_gallery(model_name, sel_idx, man_n, knn_n, knn_dist)

            # Add limit notice if needed
            if at_limit:
                info += " (max 20)"

            return (
                gr.update(),   # selected_image
                gr.update(),   # selected_details
                sel_idx,       # selected_idx (unchanged)
                man_n,         # manual_neighbors
                knn_n,         # knn_neighbors
                fig,           # umap_plot
                gallery,       # neighbor_gallery
                info,          # neighbor_info
                traj,          # trajectory_coords (preserved)
            )

        def on_clear_selection(high_class, traj, model_name):
            """Clear selection and neighbors (preserves trajectory, preview unchanged)."""
            fig = visualizer.create_umap_figure(
                model_name,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )
            return (
                None,                      # selected_image
                "Click a point to select", # selected_details
                None,                      # selected_idx
                [],                        # manual_neighbors
                [],                        # knn_neighbors
                {},                        # knn_distances
                fig,                       # umap_plot
                [],                        # neighbor_gallery
                "No neighbors selected",   # neighbor_info
            )

        def on_class_filter(class_value, sel_idx, man_n, knn_n, traj, model_name):
            """Handle class filter selection (preserves trajectory)."""
            model_data = visualizer.get_model(model_name)
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                highlighted_class=class_value,
                trajectory=traj if traj else None,
            )

            if class_value is not None and model_data and "class_label" in model_data.df.columns:
                count = (model_data.df["class_label"] == class_value).sum()
                status = f"{count} samples"
            else:
                status = ""

            return fig, class_value, status

        def on_clear_class(sel_idx, man_n, knn_n, traj, model_name):
            """Clear class highlight (preserves trajectory)."""
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n,
                knn_neighbors=knn_n,
                trajectory=traj if traj else None,
            )
            return fig, None, None, ""

        def on_model_switch(new_model_name, cur_model, _sel_idx, _man_n, _knn_n, _knn_dist, _high_class):
            """Handle model switching (resets all state including preview).

            Single-model-at-a-time: unloads current model, loads new one.
            """
            if new_model_name == cur_model:
                return (gr.update(),) * 24

            if not visualizer.is_valid_model(new_model_name):
                return (gr.update(),) * 24

            # Load new model (unloads current automatically)
            if not visualizer._ensure_model_loaded(new_model_name):
                return (gr.update(),) * 24

            model_data = visualizer.get_model(new_model_name)
            fig = visualizer.create_umap_figure(new_model_name)
            status = f"Showing {len(model_data.df)} samples ({new_model_name})"

            return (
                new_model_name,                    # current_model (session state)
                fig,                               # umap_plot
                status,                            # status_text
                f"Switched to {new_model_name}",   # model_status
                None,                              # selected_idx
                [],                                # manual_neighbors
                [],                                # knn_neighbors
                {},                                # knn_distances
                None,                              # highlighted_class
                [],                                # trajectory_coords
                None,                              # preview_image
                "Hover over a point to preview",   # preview_details
                None,                              # selected_image
                "Click a point to select",         # selected_details
                gr.update(choices=visualizer.get_class_options(new_model_name), value=None),  # class_dropdown
                None,                              # generated_image
                gr.update(value=[], label="Denoising Steps"),  # intermediate_gallery
                gr.update(choices=[], value=None),  # gen_traj_dropdown
                [],                                # intermediate_images
                -1,                                # animation_frame
                [],                                # generation_infos
                [],                                # neighbor_gallery
                "No neighbors selected",           # neighbor_info
                gr.update(choices=visualizer.get_layer_choices(new_model_name), value=visualizer.get_default_layer_label(new_model_name)),  # layer_dropdown
            )

        # Wire up events
        # on_load initializes KNN model
        app.load(on_load, outputs=[])

        # Hover handling via JS bridge (updates preview panel)
        # Gradio 6: use .change() instead of .input()
        hover_data_box.change(
            on_hover_data,
            inputs=[hover_data_box, intermediate_images, current_model],
            outputs=[preview_image, preview_details],
        )

        # Click handling via JS bridge (click_data_box receives JSON from Plotly click)
        # Gradio 6: use .change() instead of .input()
        click_data_box.change(
            on_click_data,
            inputs=[
                click_data_box, selected_idx, manual_neighbors,
                knn_neighbors, knn_distances, highlighted_class, trajectory_coords,
                current_model
            ],
            outputs=[
                selected_image,
                selected_details,
                selected_idx,
                manual_neighbors,
                knn_neighbors,
                umap_plot,
                neighbor_gallery,
                neighbor_info,
                trajectory_coords,
            ],
        )

        clear_selection_btn.click(
            on_clear_selection,
            inputs=[highlighted_class, trajectory_coords, current_model],
            outputs=[
                selected_image,
                selected_details,
                selected_idx,
                manual_neighbors,
                knn_neighbors,
                knn_distances,
                umap_plot,
                neighbor_gallery,
                neighbor_info,
            ],
        )

        class_dropdown.change(
            on_class_filter,
            inputs=[class_dropdown, selected_idx, manual_neighbors, knn_neighbors, trajectory_coords, current_model],
            outputs=[umap_plot, highlighted_class, class_status],
        )

        clear_class_btn.click(
            on_clear_class,
            inputs=[selected_idx, manual_neighbors, knn_neighbors, trajectory_coords, current_model],
            outputs=[umap_plot, highlighted_class, class_dropdown, class_status],
        )

        if len(visualizer.model_configs) > 1:
            model_dropdown.change(
                on_model_switch,
                inputs=[
                    model_dropdown, current_model, selected_idx, manual_neighbors,
                    knn_neighbors, knn_distances, highlighted_class
                ],
                outputs=[
                    current_model,
                    umap_plot,
                    status_text,
                    model_status,
                    selected_idx,
                    manual_neighbors,
                    knn_neighbors,
                    knn_distances,
                    highlighted_class,
                    trajectory_coords,
                    preview_image,
                    preview_details,
                    selected_image,
                    selected_details,
                    class_dropdown,
                    generated_image,
                    intermediate_gallery,
                    gen_traj_dropdown,
                    intermediate_images,
                    animation_frame,
                    generation_infos,
                    neighbor_gallery,
                    neighbor_info,
                    layer_dropdown,
                ],
            )

        # --- Layer change handler ---
        def on_layer_change(layer_name, model_name):
            """Handle layer dropdown change: recompute UMAP for selected layer."""
            model_data = visualizer.get_model(model_name)
            if model_data is None or not layer_name:
                return (gr.update(),) * 16

            # If user selected the default label, restore pre-computed embeddings
            default_label = visualizer.get_default_layer_label(model_name)
            if layer_name == default_label:
                visualizer._restore_default_embeddings(model_name)
                fig = visualizer.create_umap_figure(model_name)
                n = len(model_data.df)
                return (
                    fig,
                    f"Showing {n} samples ({model_name})",
                    None, [], [], {}, [], [], "No neighbors selected",
                    None, gr.update(value=[], label="Denoising Steps"),
                    gr.update(choices=[], value=None), [], [],
                    None, "Click a point to select",
                )

            success = visualizer.recompute_layer_umap(model_name, layer_name)
            if not success:
                visualizer._restore_default_embeddings(model_name)
                fig = visualizer.create_umap_figure(model_name)
                n = len(model_data.df)
                return (
                    fig,
                    f"Showing {n} samples ({model_name}) — failed {layer_name}, restored default",
                    None, [], [], {}, [], [], "No neighbors selected",
                    None, gr.update(value=[], label="Denoising Steps"),
                    gr.update(choices=[], value=None), [], [],
                    None, "Click a point to select",
                )

            fig = visualizer.create_umap_figure(model_name)
            n = len(model_data.df)
            return (
                fig,                                                            # umap_plot
                f"Showing {n} samples ({model_name}) — layer: {layer_name}",   # status_text
                None,                                                           # selected_idx
                [],                                                             # manual_neighbors
                [],                                                             # knn_neighbors
                {},                                                             # knn_distances
                [],                                                             # trajectory_coords
                [],                                                             # neighbor_gallery
                "No neighbors selected",                                        # neighbor_info
                None,                                                           # generated_image
                gr.update(value=[], label="Denoising Steps"),                   # intermediate_gallery
                gr.update(choices=[], value=None),                              # gen_traj_dropdown
                [],                                                             # intermediate_images
                [],                                                             # generation_infos
                None,                                                           # selected_image
                "Click a point to select",                                      # selected_details
            )

        layer_dropdown.change(
            on_layer_change,
            inputs=[layer_dropdown, current_model],
            outputs=[
                umap_plot,
                status_text,
                selected_idx,
                manual_neighbors,
                knn_neighbors,
                knn_distances,
                trajectory_coords,
                neighbor_gallery,
                neighbor_info,
                generated_image,
                intermediate_gallery,
                gen_traj_dropdown,
                intermediate_images,
                generation_infos,
                selected_image,
                selected_details,
            ],
        )

        # Note: neighbor display is updated directly in click/suggest handlers
        # No need for state.change() listeners which can cause duplicate events

        # --- Suggest neighbors button ---
        def on_suggest_neighbors(sel_idx, k_val, high_class, man_n, traj, model_name):
            """Auto-suggest K nearest neighbors (preserves trajectory)."""
            if sel_idx is None:
                return gr.update(), [], {}, [], "Select a point first"

            # Clamp k to max 20
            k_val = int(k_val)
            clamped = k_val > 20
            k_val = min(k_val, 20)

            # Find KNN neighbors
            neighbors = visualizer.find_knn_neighbors(model_name, sel_idx, k=k_val)
            if not neighbors:
                return gr.update(), [], {}, [], "No neighbors found"

            # Extract indices and distances
            knn_idx = [idx for idx, _ in neighbors]
            knn_dist = dict(neighbors)

            # Update plot (preserve trajectory)
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_idx,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )

            # Build neighbor gallery
            gallery, info = build_neighbor_gallery(model_name, sel_idx, man_n or [], knn_idx, knn_dist)

            # Add clamped notice if needed
            if clamped:
                info += " (max 20)"

            return fig, knn_idx, knn_dist, gallery, info

        suggest_btn.click(
            on_suggest_neighbors,
            inputs=[selected_idx, knn_k_slider, highlighted_class, manual_neighbors, trajectory_coords, current_model],
            outputs=[umap_plot, knn_neighbors, knn_distances, neighbor_gallery, neighbor_info],
        )

        # --- Clear neighbors button ---
        def on_clear_neighbors(sel_idx, high_class, traj, model_name):
            """Clear all neighbors (preserves trajectory)."""
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                highlighted_class=high_class,
                trajectory=traj if traj else None,
            )
            return fig, [], [], {}, [], "No neighbors selected"

        clear_neighbors_btn.click(
            on_clear_neighbors,
            inputs=[selected_idx, highlighted_class, trajectory_coords, current_model],
            outputs=[
                umap_plot, manual_neighbors, knn_neighbors,
                knn_distances, neighbor_gallery, neighbor_info
            ],
        )

        # --- Generate button ---
        def on_generate(
            sel_idx, man_n, knn_n, n_steps, m_steps, guidance, s_max, s_min, noise_mode,
            high_class, existing_traj, model_name, intermediates_state, gen_infos_state
        ):
            """Generate image from selected neighbors with trajectory visualization."""
            existing_traj = existing_traj or []

            # Get model data
            gen_infos_state = gen_infos_state or []
            model_data = visualizer.get_model(model_name)
            if model_data is None:
                return None, gr.update(), gr.update(), gr.update(), existing_traj, [], gen_infos_state, -1

            # Combine all neighbors
            all_neighbors = list(set((man_n or []) + (knn_n or [])))
            if sel_idx is not None and sel_idx not in all_neighbors:
                all_neighbors.insert(0, sel_idx)

            if not all_neighbors:
                return None, gr.update(), gr.update(), gr.update(), existing_traj, [], gen_infos_state, -1

            # Get class label from selected point (or first neighbor)
            ref_idx = sel_idx if sel_idx is not None else all_neighbors[0]
            if "class_label" in model_data.df.columns:
                class_label = int(model_data.df.iloc[ref_idx]["class_label"])
            else:
                class_label = None

            # Get layers for trajectory extraction
            extract_layers = sorted(model_data.umap_params.get("layers", []))
            can_project = model_data.umap_reducer is not None and len(extract_layers) > 0

            # Run generation on GPU
            result = _generate_on_gpu(
                model_name, all_neighbors, class_label,
                n_steps, m_steps, s_max, s_min, guidance, noise_mode,
                extract_layers, can_project
            )
            if result is None:
                return None, gr.update(), gr.update(), gr.update(), [], [], gen_infos_state, -1

            # Unpack results: (images, labels, [trajectory], [intermediates], [noised_inputs])
            images = result[0]
            trajectory_acts = []
            intermediate_imgs = []
            noised_inputs = []
            idx = 2  # Start after images, labels
            if can_project:
                trajectory_acts = result[idx] if len(result) > idx else []
                idx += 1
            intermediate_imgs = result[idx] if len(result) > idx else []
            idx += 1
            noised_inputs = result[idx] if len(result) > idx else []

            # Compute sigma schedule used during this generation (store with results)
            rho = 7.0
            sigmas = []
            for i in range(int(n_steps)):
                ramp = i / max(int(n_steps) - 1, 1)
                min_inv_rho = float(s_min) ** (1 / rho)
                max_inv_rho = float(s_max) ** (1 / rho)
                sigma = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
                sigmas.append(sigma)

            # Project trajectory through UMAP
            traj_coords = []
            if trajectory_acts and model_data.umap_reducer:
                # Pin UMAP random_state for deterministic transform
                model_data.umap_reducer.random_state = 42
                for i, act in enumerate(trajectory_acts):
                    try:
                        # Scale if scaler exists
                        if model_data.umap_scaler is not None:
                            act = model_data.umap_scaler.transform(act)
                        # PCA pre-reduction if used during fitting
                        if model_data.umap_pca is not None:
                            act = model_data.umap_pca.transform(act)
                        # Project to 2D
                        coords = model_data.umap_reducer.transform(act)
                        sigma = sigmas[i] if i < len(sigmas) else 0.0
                        traj_coords.append((float(coords[0, 0]), float(coords[0, 1]), sigma))
                    except Exception as e:
                        print(f"[Trajectory] Failed to project step {i}: {e}")

            # Append new trajectory to existing list
            all_trajectories = list(existing_traj)
            if traj_coords:
                all_trajectories.append(traj_coords)

            # Build updated plot with all trajectories
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_n or [],
                highlighted_class=high_class,
                trajectory=all_trajectories if all_trajectories else None,
            )

            # Convert to numpy for gr.Image
            gen_img_raw = images[0].numpy()
            class_name = visualizer.get_class_name(class_label) if class_label else "random"
            n_steps = len(intermediate_imgs)

            # Create composite for final image (output + last noised input as inset)
            if noised_inputs and len(noised_inputs) > 0:
                last_noised = noised_inputs[-1][0].numpy()
                gen_img = GradioVisualizer.create_composite_image(gen_img_raw, last_noised)
            else:
                gen_img = gen_img_raw

            # Build generation info for this trajectory
            gen_info = {
                "class_id": class_label,
                "class_name": class_name,
                "n_steps": n_steps,
                "final_image": gen_img,
            }
            gen_infos_state = list(gen_infos_state)
            gen_infos_state.append(gen_info)

            # Build intermediate gallery and state: list of (image, sigma) tuples
            # Each step shows denoised output with noised input as inset
            step_gallery = []
            intermediates_state.append([])  # For trajectory hover
            for i, step_img in enumerate(intermediate_imgs):
                sigma = sigmas[i] if i < len(sigmas) else 0.0
                img_np = step_img[0].numpy()

                # Create composite with noised input inset if available
                if noised_inputs and i < len(noised_inputs):
                    noised_np = noised_inputs[i][0].numpy()
                    composite_img = GradioVisualizer.create_composite_image(img_np, noised_np)
                else:
                    composite_img = img_np

                caption = f"{class_label}: {class_name} | Step {i+1}/{n_steps} | σ={sigma:.1f}"
                step_gallery.append((composite_img, caption))
                intermediates_state[-1].append((composite_img, sigma))

            # Gallery label
            gallery_label = f"{class_label}: {class_name} | {n_steps} steps"
            gallery_update = gr.update(value=step_gallery, label=gallery_label)

            # Build dropdown choices: (label, index) for each trajectory
            traj_choices = []
            for i, info in enumerate(gen_infos_state):
                cid = info.get("class_id", "?")
                cname = info.get("class_name", "")
                traj_choices.append((f"Traj {i+1} Class {cid}: {cname}", i))
            new_traj_idx = len(gen_infos_state) - 1
            dropdown_update = gr.update(choices=traj_choices, value=new_traj_idx)

            return (
                gen_img, gallery_update, dropdown_update, fig,
                all_trajectories, intermediates_state, gen_infos_state, -1
            )

        generate_btn.click(
            on_generate,
            inputs=[
                selected_idx, manual_neighbors, knn_neighbors,
                num_steps_slider, mask_steps_slider, guidance_slider,
                sigma_max_input, sigma_min_input, noise_mode_dropdown,
                highlighted_class, trajectory_coords, current_model,
                intermediate_images, generation_infos
            ],
            outputs=[
                generated_image, intermediate_gallery, gen_traj_dropdown,
                umap_plot, trajectory_coords, intermediate_images,
                generation_infos, animation_frame
            ],
        )

        # --- Clear generated button ---
        def on_clear_generated(sel_idx, man_n, knn_n, high_class, model_name):
            """Clear generated image, intermediates, and trajectory."""
            fig = visualizer.create_umap_figure(
                model_name,
                selected_idx=sel_idx,
                manual_neighbors=man_n or [],
                knn_neighbors=knn_n or [],
                highlighted_class=high_class,
            )
            gallery_update = gr.update(value=[], label="Denoising Steps")
            dropdown_update = gr.update(choices=[], value=None)
            return None, gallery_update, dropdown_update, fig, [], [], -1, []

        clear_gen_btn.click(
            on_clear_generated,
            inputs=[selected_idx, manual_neighbors, knn_neighbors, highlighted_class, current_model],
            outputs=[
                generated_image, intermediate_gallery, gen_traj_dropdown,
                umap_plot, trajectory_coords, intermediate_images,
                animation_frame, generation_infos
            ],
        )

        # --- Frame navigation for intermediate images ---
        def format_frame_info(gen_info, frame_idx, n_frames, sigma):
            """Format frame info string with class, step, sigma (compact)."""
            if not gen_info:
                return f"Step {frame_idx + 1}/{n_frames} | σ={sigma:.1f}"

            class_id = gen_info.get("class_id", "?")
            class_name = gen_info.get("class_name", "")

            return f"{class_id}: {class_name} | Step {frame_idx + 1}/{n_frames} | σ={sigma:.1f}"

        def _get_traj_idx(traj_sel, intermediates):
            """Resolve selected trajectory index, default to last."""
            if traj_sel is not None and isinstance(traj_sel, int):
                return traj_sel
            return len(intermediates) - 1 if intermediates else -1

        def _get_gen_info(traj_idx, gen_infos):
            """Get gen_info dict for a trajectory index."""
            if gen_infos and 0 <= traj_idx < len(gen_infos):
                return gen_infos[traj_idx]
            return None

        def on_next_frame(intermediates, current_frame, gen_infos, traj_sel):
            """Show next intermediate frame."""
            ti = _get_traj_idx(traj_sel, intermediates)
            if ti < 0 or not intermediates[ti]:
                return gr.update(), -1, gr.update()

            n_frames = len(intermediates[ti])
            if current_frame == -1:
                new_frame = 0
            else:
                new_frame = (current_frame + 1) % n_frames

            img, sigma = intermediates[ti][new_frame]
            label = format_frame_info(_get_gen_info(ti, gen_infos), new_frame, n_frames, sigma)
            return img, new_frame, gr.update(label=label)

        def on_prev_frame(intermediates, current_frame, gen_infos, traj_sel):
            """Show previous intermediate frame."""
            ti = _get_traj_idx(traj_sel, intermediates)
            if ti < 0 or not intermediates[ti]:
                return gr.update(), -1, gr.update()

            n_frames = len(intermediates[ti])
            if current_frame <= 0:
                new_frame = n_frames - 1
            else:
                new_frame = current_frame - 1

            img, sigma = intermediates[ti][new_frame]
            label = format_frame_info(_get_gen_info(ti, gen_infos), new_frame, n_frames, sigma)
            return img, new_frame, gr.update(label=label)

        next_frame_btn.click(
            on_next_frame,
            inputs=[intermediate_images, animation_frame, generation_infos, gen_traj_dropdown],
            outputs=[generated_image, animation_frame, intermediate_gallery],
        )

        prev_frame_btn.click(
            on_prev_frame,
            inputs=[intermediate_images, animation_frame, generation_infos, gen_traj_dropdown],
            outputs=[generated_image, animation_frame, intermediate_gallery],
        )

        # Clicking gallery item shows it in main image
        def on_gallery_select(evt: gr.SelectData, intermediates, gen_infos, traj_sel):
            """Show selected gallery item in main generated image."""
            ti = _get_traj_idx(traj_sel, intermediates)
            if ti < 0 or not intermediates[ti] or evt.index >= len(intermediates[ti]):
                return gr.update(), -1, gr.update()

            img, sigma = intermediates[ti][evt.index]
            n_frames = len(intermediates[ti])
            label = format_frame_info(_get_gen_info(ti, gen_infos), evt.index, n_frames, sigma)
            return img, evt.index, gr.update(label=label)

        intermediate_gallery.select(
            on_gallery_select,
            inputs=[intermediate_images, generation_infos, gen_traj_dropdown],
            outputs=[generated_image, animation_frame, intermediate_gallery],
        )

        # --- Trajectory selector dropdown ---
        def on_traj_select(traj_sel, intermediates, gen_infos):
            """Switch displayed trajectory when dropdown changes."""
            if traj_sel is None or not intermediates or not gen_infos:
                return gr.update(), gr.update(), -1

            ti = int(traj_sel)
            if ti < 0 or ti >= len(intermediates) or ti >= len(gen_infos):
                return gr.update(), gr.update(), -1

            info = gen_infos[ti]
            final_img = info.get("final_image")

            # Rebuild gallery for this trajectory
            steps = intermediates[ti]
            step_gallery = []
            cid = info.get("class_id", "?")
            cname = info.get("class_name", "")
            n_steps = len(steps)
            for i, (img, sigma) in enumerate(steps):
                caption = f"{cid}: {cname} | Step {i+1}/{n_steps} | σ={sigma:.1f}"
                step_gallery.append((img, caption))

            gallery_label = f"{cid}: {cname} | {n_steps} steps"
            gallery_update = gr.update(value=step_gallery, label=gallery_label)

            return final_img, gallery_update, -1

        gen_traj_dropdown.change(
            on_traj_select,
            inputs=[gen_traj_dropdown, intermediate_images, generation_infos],
            outputs=[generated_image, intermediate_gallery, animation_frame],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio Diffusion Activation Visualizer")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to embeddings CSV")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server on")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device (auto-detected if not specified)",
    )
    parser.add_argument("--num-steps", type=int, default=5, help="Number of denoising steps")
    parser.add_argument("--mask-steps", type=int, default=1, help="Steps to apply mask")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--sigma-max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--sigma-min", type=float, default=0.5, help="Minimum sigma")
    parser.add_argument("--max-classes", "-c", type=int, default=None, help="Max classes to load")
    parser.add_argument("--model", "-m", type=str, default=None, help="Initial model to load")
    args = parser.parse_args()

    visualizer = GradioVisualizer(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings,
        device=get_device(args.device),
        num_steps=args.num_steps,
        mask_steps=args.mask_steps,
        guidance_scale=args.guidance_scale,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        max_classes=args.max_classes,
        initial_model=args.model,
    )

    app = create_gradio_app(visualizer)
    app.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        js=PLOTLY_HANDLER_JS,
    )


if __name__ == "__main__":
    main()

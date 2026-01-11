"""
Interactive Dash application for DMD2 activation visualization.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import json
import argparse
import pickle
import torch
from sklearn.neighbors import NearestNeighbors

# For loading dataset activations (UMAP pre-computed via CLI or embeddings CSV)
from diffviews.processing.umap import load_dataset_activations

# For generation from activations - using diffviews adapter interface
from diffviews.adapters import get_adapter
from diffviews.core.masking import ActivationMasker, unflatten_activation
from diffviews.core.generator import (
    generate_with_mask,
    generate_with_mask_multistep,
    get_denoising_sigmas,
    save_generated_sample
)


class DMD2Visualizer:
    """Main visualizer application."""

    def __init__(self, data_dir: Path, embeddings_path: Path = None, checkpoint_path: Path = None,
                 device: str = 'cuda', num_steps: int = 1, mask_steps: int = None,
                 guidance_scale: float = 1.0, sigma_max: float = 80.0, sigma_min: float = 0.002,
                 label_dropout: float = 0.0, adapter_name: str = 'dmd2-imagenet-64',
                 umap_n_neighbors: int = 15, umap_min_dist: float = 0.1, max_classes: int = None):
        """
        Args:
            data_dir: Root data directory
            embeddings_path: Optional path to precomputed embeddings CSV
            checkpoint_path: Optional path to checkpoint for generation
            device: Device for generation ('cuda', 'mps', or 'cpu')
            num_steps: Number of denoising steps (1=single-step, 4/10=multi-step)
            mask_steps: Steps to apply activation mask (default=num_steps, 1=first-step-only)
            guidance_scale: CFG scale (0=uncond, 1=class, >1=amplify, <0=anti-class)
            sigma_max: Maximum sigma for denoising schedule
            sigma_min: Minimum sigma for denoising schedule
            label_dropout: Label dropout for model config (use 0.1 for CFG models)
            adapter_name: Adapter name for model loading (default: dmd2-imagenet-64)
            umap_n_neighbors: UMAP n_neighbors parameter
            umap_min_dist: UMAP min_dist parameter
            max_classes: Maximum number of classes to load (None=all)
        """
        self.data_dir = Path(data_dir)
        self.embeddings_path = embeddings_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_steps = num_steps
        self.mask_steps = mask_steps
        self.guidance_scale = guidance_scale
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.label_dropout = label_dropout
        self.adapter_name = adapter_name
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.max_classes = max_classes
        self.df = None
        self.umap_params = None
        self.activations = None
        self.metadata_df = None
        self.nn_model = None  # Nearest neighbors model
        self.selected_point = None  # Currently selected point
        self.neighbor_indices = None  # Indices of neighbors
        self.class_labels = {}  # ImageNet class labels (Standard ordering)

        # For generation from activations
        self.umap_reducer = None  # UMAP model for inverse_transform
        self.umap_scaler = None   # Scaler for inverse_transform
        self.adapter = None       # GeneratorAdapter instance
        self.layer_shapes = {}    # Cache of layer activation shapes

        # Load class labels
        self.load_class_labels()

        # Load data
        print(f"Data directory: {self.data_dir.absolute()}")
        self.load_data()

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.app.title = "DMD2 Activation Visualizer"
        self.build_layout()
        self.register_callbacks()

    def load_class_labels(self):
        """Load ImageNet class labels (Standard ordering)."""
        # All data now uses Standard ImageNet ordering - no remapping needed
        label_path = self.data_dir / "imagenet_standard_class_index.json"
        if not label_path.exists():
            # Fallback to legacy paths
            label_path = self.data_dir / "imagenet_class_labels.json"
        if not label_path.exists():
            # Fallback to package data directory
            label_path = Path(__file__).parent.parent / "data" / "imagenet_standard_class_index.json"

        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                raw_labels = json.load(f)
                # Convert to dict: {class_id: class_name}
                self.class_labels = {int(k): v[1] for k, v in raw_labels.items()}
            print(f"Loaded {len(self.class_labels)} ImageNet class labels (Standard ordering)")
        else:
            print(f"Warning: Class labels not found at {label_path}")
            self.class_labels = {}

    def get_class_name(self, class_id):
        """Get human-readable class name for a class ID."""
        if class_id in self.class_labels:
            return self.class_labels[class_id]
        return f"Unknown class {class_id}"

    def get_sample_class_name(self, sample):
        """Get class name from sample, handling NaN values."""
        if 'class_label' not in sample:
            return None
        class_id = int(sample['class_label'])
        class_name = sample.get('class_name')
        if class_name is None or (isinstance(class_name, float) and pd.isna(class_name)):
            class_name = self.get_class_name(class_id)
        return class_name

    def add_class_highlight_trace(self, fig, highlighted_class):
        """Add class highlight X markers colored by log(sigma)."""
        if highlighted_class is None or 'class_label' not in self.df.columns:
            return

        class_mask = self.df['class_label'] == highlighted_class
        class_df = self.df[class_mask]

        if len(class_df) == 0:
            return

        class_name = self.get_class_name(highlighted_class)

        # Color by log(sigma): black at max, medium grey at min
        # Filter to only samples with valid sigma values
        if 'conditioning_sigma' in class_df.columns:
            sigma_mask = class_df['conditioning_sigma'].notna()
            class_df_with_sigma = class_df[sigma_mask]

            if len(class_df_with_sigma) > 0:
                sigmas = class_df_with_sigma['conditioning_sigma'].values
                log_sigmas = np.log(sigmas + 1e-6)  # Avoid log(0)
                # Normalize to 0-1 range (0=min sigma, 1=max sigma)
                log_min, log_max = log_sigmas.min(), log_sigmas.max()
                if log_max > log_min:
                    normalized = (log_sigmas - log_min) / (log_max - log_min)
                else:
                    normalized = np.ones_like(log_sigmas) * 0.5
                # Map: 1 (max sigma) -> black (0), 0 (min sigma) -> medium grey (128)
                grey_values = ((1 - normalized) * 128).astype(int)
                colors = [f'rgb({g},{g},{g})' for g in grey_values]

                # Add trace for samples with sigma (colored)
                fig.add_trace(go.Scatter(
                    x=class_df_with_sigma['umap_x'],
                    y=class_df_with_sigma['umap_y'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors,
                        symbol='x',
                        line=dict(width=2, color=colors)
                    ),
                    name=f'Class {highlighted_class}: {class_name}',
                    hoverinfo='skip',
                    showlegend=True
                ))

                # Add separate trace for samples without sigma (generated) in green
                class_df_no_sigma = class_df[~sigma_mask]
                if len(class_df_no_sigma) > 0:
                    fig.add_trace(go.Scatter(
                        x=class_df_no_sigma['umap_x'],
                        y=class_df_no_sigma['umap_y'],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='#00FF00',
                            symbol='x',
                            line=dict(width=2, color='#00FF00')
                        ),
                        name=f'Class {highlighted_class} (generated)',
                        hoverinfo='skip',
                        showlegend=False
                    ))
                return

        # Fallback: no sigma data, use black
        fig.add_trace(go.Scatter(
            x=class_df['umap_x'],
            y=class_df['umap_y'],
            mode='markers',
            marker=dict(
                size=12,
                color='black',
                symbol='x',
                line=dict(width=2, color='black')
            ),
            name=f'Class {highlighted_class}: {class_name}',
            hoverinfo='skip',
            showlegend=True
        ))

    def load_data(self):
        """Load embeddings or prepare for generation."""
        if self.embeddings_path:
            if not Path(self.embeddings_path).exists():
                print(f"ERROR: Embeddings file not found: {self.embeddings_path}")
                print(f"  Absolute path: {Path(self.embeddings_path).absolute()}")
                self.df = pd.DataFrame()
                return
            print(f"Loading embeddings from {self.embeddings_path}")
            self.df = pd.read_csv(self.embeddings_path)

            # Load UMAP params
            param_path = Path(self.embeddings_path).with_suffix('.json')
            if param_path.exists():
                with open(param_path, 'r', encoding='utf-8') as f:
                    self.umap_params = json.load(f)
            else:
                self.umap_params = {}

            # Load UMAP model for inverse_transform (optional)
            model_path = Path(self.embeddings_path).with_suffix('.pkl')
            if model_path.exists():
                print(f"Loading UMAP model from {model_path}")
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.umap_reducer = model_data['reducer']
                    self.umap_scaler = model_data['scaler']
                print("UMAP model loaded (inverse_transform available)")
            else:
                print(f"Note: UMAP pkl not found at {model_path}")
                print("(Generation still works - uses neighbor averaging)")

            # Filter by max_classes if specified
            if self.max_classes is not None and 'class_label' in self.df.columns:
                unique_classes = self.df['class_label'].unique()
                if len(unique_classes) > self.max_classes:
                    classes_to_keep = unique_classes[:self.max_classes]
                    original_count = len(self.df)
                    self.df = self.df[self.df['class_label'].isin(classes_to_keep)].reset_index(drop=True)
                    print(f"Filtered to {self.max_classes} classes: {original_count} -> {len(self.df)} samples")

            # Always load activations for generation
            if self.activations is None:
                self.activations, self.metadata_df = self.load_activations_for_model("imagenet_real")

            print(f"Loaded {len(self.df)} samples")
        else:
            print("No embeddings found. Will load activations for dynamic UMAP.")
            self.df = pd.DataFrame()

    def load_activations_for_model(self, model_type: str):
        """Load raw activations for dynamic UMAP computation."""
        activation_dir = self.data_dir / "activations" / model_type
        metadata_path = self.data_dir / "metadata" / model_type / "dataset_info.json"

        if not activation_dir.exists():
            print(f"Warning: Activation dir not found: {activation_dir}")
            print(f"  Expected structure: {self.data_dir}/activations/{model_type}/")
            return None, None

        if not metadata_path.exists():
            print(f"Warning: No metadata found: {metadata_path}")
            print(f"  Expected: {self.data_dir}/metadata/{model_type}/dataset_info.json")
            return None, None

        activations, metadata_df = load_dataset_activations(
            activation_dir,
            metadata_path
        )
        return activations, metadata_df

    def get_image_base64(self, image_path: str, size: tuple = (256, 256)):
        """Convert image to base64 for hover display."""
        try:
            full_path = self.data_dir / image_path
            img = Image.open(full_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def build_layout(self):
        """Build Dash layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("DMD2 Activation Visualizer", className="mb-4")
                ])
            ]),

            dbc.Row([
                # Left sidebar - controls + hover preview
                dbc.Col([
                    # Controls card
                    dbc.Card([
                        dbc.CardHeader("Controls"),
                        dbc.CardBody([
                            # Export button
                            dbc.Button(
                                "Export Data",
                                id="export-btn",
                                color="secondary",
                                className="w-100",
                                disabled=True
                            ),
                            dcc.Download(id="download-data"),

                            # Status
                            html.Hr(),
                            html.Div(id="status-text", className="text-muted small")
                        ])
                    ], className="mb-3"),

                    # Hover preview card
                    dbc.Card([
                        dbc.CardHeader("Hover Preview"),
                        dbc.CardBody([
                            html.Div(id="hover-image"),
                            html.Div(id="hover-details", className="mt-2")
                        ])
                    ], className="mb-3"),

                    # Class filter card
                    dbc.Card([
                        dbc.CardHeader("Find Class"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="class-filter-dropdown",
                                options=[],  # Populated dynamically
                                placeholder="Select a class...",
                                searchable=True,
                                clearable=True,
                                className="mb-2",
                                style={"maxHeight": "300px"}
                            ),
                            dbc.Button(
                                "Clear Highlight",
                                id="clear-class-highlight-btn",
                                color="secondary",
                                size="sm",
                                outline=True,
                                className="w-100",
                                disabled=True
                            ),
                            html.Div(id="class-filter-status", className="text-muted small mt-2")
                        ])
                    ])
                ], width=3),

                # Main visualization
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-plot",
                                children=[
                                    dcc.Graph(
                                        id="umap-scatter",
                                        style={"height": "70vh"}
                                    )
                                ],
                                type="default"
                            )
                        ])
                    ])
                ], width=6),

                # Right sidebar - selected sample + neighbors
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Selected Sample"),
                            dbc.Button(
                                "✕",
                                id="clear-selection-btn",
                                color="link",
                                size="sm",
                                className="float-end p-0",
                                style={"fontSize": "20px", "lineHeight": "1", "display": "none"}
                            )
                        ]),
                        dbc.CardBody([
                            html.Div(id="selected-image"),
                            html.Div(id="selected-details", className="mt-2"),
                            html.Hr(),

                            # Generation controls
                            html.Label("Generation Settings", className="fw-bold"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Steps", className="small", id="steps-label"),
                                    dbc.Tooltip(
                                        "Number of denoising steps. 1=single-step, 4-10=multi-step for higher quality.",
                                        target="steps-label", placement="top"
                                    ),
                                    dbc.Input(
                                        id="num-steps-input",
                                        type="number",
                                        min=1,
                                        max=50,
                                        step=1,
                                        value=self.num_steps,
                                        size="sm"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Mask Steps", className="small", id="mask-steps-label"),
                                    dbc.Tooltip(
                                        "Steps to apply activation mask. Default=all steps. Use 1 for first-step-only masking.",
                                        target="mask-steps-label", placement="top"
                                    ),
                                    dbc.Input(
                                        id="mask-steps-input",
                                        type="number",
                                        min=1,
                                        max=50,
                                        step=1,
                                        value=self.mask_steps or self.num_steps,
                                        size="sm"
                                    ),
                                ], width=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Guidance", className="small", id="guidance-label"),
                                    dbc.Tooltip(
                                        "CFG scale. 0=unconditional, 1=class-conditional, >1=amplified, <0=negative guidance.",
                                        target="guidance-label", placement="top"
                                    ),
                                    dbc.Input(
                                        id="guidance-scale-input",
                                        type="number",
                                        min=-10,
                                        max=20,
                                        step=0.1,
                                        value=self.guidance_scale,
                                        size="sm"
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("σ max", className="small", id="sigma-max-label"),
                                    dbc.Tooltip(
                                        "Maximum noise level (start of denoising). Higher=more noise.",
                                        target="sigma-max-label", placement="top"
                                    ),
                                    dbc.Input(
                                        id="sigma-max-input",
                                        type="number",
                                        min=0.01,
                                        max=200,
                                        step=1,
                                        value=self.sigma_max,
                                        size="sm"
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("σ min", className="small", id="sigma-min-label"),
                                    dbc.Tooltip(
                                        "Minimum noise level (end of denoising). Lower=cleaner output.",
                                        target="sigma-min-label", placement="top"
                                    ),
                                    dbc.Input(
                                        id="sigma-min-input",
                                        type="number",
                                        min=0.0001,
                                        max=10,
                                        step=0.001,
                                        value=self.sigma_min,
                                        size="sm"
                                    ),
                                ], width=4),
                            ], className="mb-2"),

                            html.Hr(),
                            html.Label("Generate from Neighbors"),
                            dbc.Button(
                                "Generate Image",
                                id="generate-from-neighbors-btn",
                                color="success",
                                size="sm",
                                className="w-100 mb-2",
                                disabled=True
                            ),
                            html.Div(id="generation-status", className="text-muted small mb-2"),
                            dbc.Button(
                                "Clear Generated",
                                id="clear-generated-btn",
                                color="secondary",
                                size="sm",
                                outline=True,
                                className="w-100 mb-2",
                            ),

                            html.Hr(),
                            html.Label("Manual Neighbor Selection"),
                            html.Div(
                                "Click on other points to add/remove neighbors",
                                className="text-muted small mb-2"
                            ),
                            html.Div(id="neighbor-list", className="small", style={"maxHeight": "400px", "overflowY": "auto"})
                        ])
                    ])
                ], width=3)
            ]),

            # Hidden stores for state
            dcc.Store(id="selected-point-store", data=None),
            dcc.Store(id="neighbor-indices-store", data=None),
            dcc.Store(id="manual-neighbors-store", data=[]),
            dcc.Store(id="highlighted-class-store", data=None)
        ], fluid=True, className="p-4")

    def fit_nearest_neighbors(self):
        """Fit KNN model on UMAP coordinates (2D) for intuitive neighbor selection."""
        if self.df.empty or 'umap_x' not in self.df.columns:
            print("[KNN] No UMAP coordinates loaded, skipping KNN fit")
            return

        # Fit on 2D UMAP coordinates for intuitive visual neighbor selection
        umap_coords = self.df[['umap_x', 'umap_y']].values
        print(f"[KNN] Fitting on UMAP coordinates: {umap_coords.shape}")
        self.nn_model = NearestNeighbors(n_neighbors=21, metric='euclidean')
        self.nn_model.fit(umap_coords)

    def register_callbacks(self):
        """Register Dash callbacks."""

        @self.app.callback(
            Output("umap-scatter", "figure"),
            Output("status-text", "children"),
            Input("status-text", "id"),  # Dummy input to trigger on load
            prevent_initial_call=False
        )
        def update_plot(_):
            """Display pre-loaded UMAP embeddings."""
            # Fit NN model if not already done
            if self.nn_model is None and not self.df.empty:
                self.fit_nearest_neighbors()

            # Create plot
            if self.df.empty:
                return go.Figure(), "No data loaded"

            # Determine color column
            if 'class_label' in self.df.columns:
                color_col = 'class_label'
                hover_data = ['class_label']
            else:
                color_col = None
                hover_data = []

            fig = px.scatter(
                self.df,
                x='umap_x',
                y='umap_y',
                color=color_col,
                hover_data=hover_data + ['sample_id'],
                title="DMD2 Activation UMAP",
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2'}
            )

            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(
                hovermode='closest',
                template='plotly_white'
            )

            status = f"Showing {len(self.df)} samples"
            return fig, status

        @self.app.callback(
            Output("hover-image", "children"),
            Output("hover-details", "children"),
            Input("umap-scatter", "hoverData")
        )
        def display_hover(hoverData):
            """Display image and info on hover."""
            if not hoverData or self.df.empty:
                return "Hover over a point", html.Div("No point hovered", className="text-muted small")

            point_data = hoverData['points'][0]
            curve_number = point_data.get('curveNumber', 0)

            # If hovering over non-main trace with customdata, check type to distinguish
            # Trajectory points: customdata[0] is string (sample_id)
            # Generated overlay: customdata[0] is int (df index)
            if curve_number > 0 and 'customdata' in point_data:
                first_elem = point_data['customdata'][0]

                # Trajectory point: customdata = [sample_id_str, step_str, img_path]
                if isinstance(first_elem, str):
                    customdata = point_data['customdata']
                    sample_id, step, img_path = customdata[0], customdata[1], customdata[2]

                    # Look up full trajectory for this sample
                    full_trajectory = None
                    if hasattr(self, 'generated_trajectories'):
                        for traj in self.generated_trajectories:
                            if traj['sample_id'] == sample_id:
                                full_trajectory = traj.get('trajectory', [])
                                break

                    # Build grid of all trajectory images with hovered one highlighted
                    if full_trajectory:
                        grid_items = []
                        for traj_point in full_trajectory:
                            traj_step_num = traj_point['step']
                            traj_img_path = traj_point.get('image_path')
                            traj_sigma = traj_point.get('sigma')
                            # Compare step numbers (step string may include sigma suffix like "step 0, σ=80.0")
                            step_prefix = f"step {traj_step_num}"
                            is_hovered = (step == step_prefix or step.startswith(step_prefix + ","))

                            # Format label with sigma if available
                            if traj_sigma is not None:
                                step_label = f"Step {traj_point['step']}, σ={traj_sigma:.3f}"
                            else:
                                step_label = f"Step {traj_point['step']}"

                            if traj_img_path:
                                img_b64 = self.get_image_base64(traj_img_path, size=(150, 150))
                                border_style = "3px solid red" if is_hovered else "1px solid #ccc"
                                opacity = "1" if is_hovered else "0.7"
                                grid_items.append(html.Div([
                                    html.Img(
                                        src=img_b64,
                                        style={"width": "100%", "border": border_style, "borderRadius": "4px", "opacity": opacity}
                                    ) if img_b64 else html.Div("?", style={"width": "60px", "height": "60px"}),
                                    html.Div(step_label, className="text-center small",
                                             style={"fontWeight": "bold" if is_hovered else "normal"})
                                ], style={"width": "calc(50% - 4px)", "margin": "2px", "textAlign": "center"}))

                        img_element = html.Div(grid_items, style={
                            "display": "flex", "flexWrap": "wrap", "justifyContent": "center"
                        })
                    elif img_path:
                        # Fallback to single image if trajectory not found
                        img_b64 = self.get_image_base64(img_path, size=(200, 200))
                        img_element = html.Img(
                            src=img_b64,
                            style={"width": "100%", "border": "2px solid red", "borderRadius": "4px"}
                        ) if img_b64 else html.Div("Image not found")
                    else:
                        img_element = html.Div("No image (intended point)", className="text-muted")

                    details = [
                        html.Div([html.Strong("Sample: "), html.Span(sample_id, className="small")]),
                        html.Div([html.Strong("Hovered: "), html.Span(step, className="small")]),
                        html.Div([html.Strong("Coords: "), html.Span(f"({point_data['x']:.3f}, {point_data['y']:.3f})", className="small")])
                    ]
                    return img_element, html.Div(details, className="small")

                # Generated overlay: customdata = [int_df_index]
                point_idx = first_elem
            else:
                # Main scatter plot (trace 0), use pointIndex directly
                point_idx = point_data['pointIndex']

            sample = self.df.iloc[point_idx]

            # Get image thumbnail for hover
            img_b64 = self.get_image_base64(sample['image_path'], size=(200, 200))
            img_element = html.Img(
                src=img_b64,
                style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "4px"}
            ) if img_b64 else html.Div("Image not found")

            # Compact details for hover
            details = []
            details.append(html.Div([
                html.Strong("ID: "),
                html.Span(sample['sample_id'], className="small")
            ]))

            class_name = self.get_sample_class_name(sample)
            if class_name is not None:
                class_id = int(sample['class_label'])
                details.append(html.Div([
                    html.Strong("Class: "),
                    html.Span(f"{class_id}: {class_name}", className="small")
                ]))

            details.append(html.Div([
                html.Strong("Coords: "),
                html.Span(f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})", className="small")
            ]))

            if 'conditioning_sigma' in sample and pd.notna(sample['conditioning_sigma']):
                details.append(html.Div([
                    html.Strong("Sigma: "),
                    html.Span(f"{sample['conditioning_sigma']}", className="small")
                ]))

            return img_element, html.Div(details)

        @self.app.callback(
            Output("selected-image", "children"),
            Output("selected-details", "children"),
            Output("generate-from-neighbors-btn", "disabled"),
            Output("selected-point-store", "data"),
            Output("manual-neighbors-store", "data"),
            Output("clear-selection-btn", "style"),
            Output("neighbor-indices-store", "data", allow_duplicate=True),
            Input("umap-scatter", "clickData"),
            Input("clear-selection-btn", "n_clicks"),
            State("selected-point-store", "data"),
            State("manual-neighbors-store", "data"),
            State("neighbor-indices-store", "data"),
            prevent_initial_call=True
        )
        def display_selected(clickData, clear_clicks, current_selected, manual_neighbors, knn_neighbors):
            """Handle point selection and neighbor toggling."""
            ctx = callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Handle clear button
            if trigger_id == "clear-selection-btn":
                return (
                    "Click a point to select",
                    html.Div("No point selected", className="text-muted small"),
                    True,  # Disable generate button
                    None,  # Clear selected point
                    [],    # Clear manual neighbors
                    {"fontSize": "20px", "lineHeight": "1", "display": "none"},  # Hide clear button
                    None   # Clear KNN neighbors
                )

            # Handle click on plot
            if not clickData or self.df.empty:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            point_data = clickData['points'][0]
            curve_number = point_data.get('curveNumber', 0)

            # If clicked on non-main trace with customdata, check type
            if curve_number > 0 and 'customdata' in point_data:
                first_elem = point_data['customdata'][0]
                # Trajectory point (string customdata) - ignore clicks
                if isinstance(first_elem, str):
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
                # Generated overlay (int customdata)
                point_idx = first_elem
            else:
                point_idx = point_data['pointIndex']

            # Ensure lists are initialized
            if manual_neighbors is None:
                manual_neighbors = []
            if knn_neighbors is None:
                knn_neighbors = []

            # If no point currently selected, select this one
            if current_selected is None:
                sample = self.df.iloc[point_idx]
                img_b64 = self.get_image_base64(sample['image_path'])
                img_element = html.Img(
                    src=img_b64,
                    style={"width": "100%", "border": "2px solid #0d6efd", "borderRadius": "4px"}
                ) if img_b64 else html.Div("Image not found")

                details = []
                details.append(html.P([html.Strong("Sample ID: "), sample['sample_id']]))
                class_name = self.get_sample_class_name(sample)
                if class_name is not None:
                    class_id = int(sample['class_label'])
                    details.append(html.P([html.Strong("Class: "), f"{class_id}: {class_name}"]))
                if 'conditioning_sigma' in sample and pd.notna(sample['conditioning_sigma']):
                    details.append(html.P([html.Strong("Sigma: "), f"{sample['conditioning_sigma']}"]))
                details.append(html.P([
                    html.Strong("UMAP Coords: "),
                    f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"
                ]))
                details.append(html.P(
                    "Click points to add or remove neighbors",
                    className="text-info small"
                ))

                # Enable generate button only if we have checkpoint
                # (umap_reducer is optional - only used for diagnostic logging)
                generate_enabled = self.checkpoint_path is not None

                return (
                    img_element,
                    html.Div(details),
                    not generate_enabled,  # Enable generate if checkpoint available
                    point_idx,
                    [],     # Reset manual neighbors
                    {"fontSize": "20px", "lineHeight": "1", "display": "inline"},  # Show clear button
                    None    # Clear KNN neighbors
                )

            # If clicking the same point, do nothing
            if point_idx == current_selected:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            # Toggle neighbor: check if in manual or KNN list
            # Priority: if in manual list, remove from manual; if in KNN list, move to manual (for removal); if in neither, add to manual
            if point_idx in manual_neighbors:
                # Remove from manual list
                manual_neighbors.remove(point_idx)
                new_knn = knn_neighbors  # Keep KNN unchanged
            elif point_idx in knn_neighbors:
                # If clicking a KNN neighbor, remove it from KNN list and add to manual remove list
                # This effectively removes it since we filter KNN against manual in display
                new_knn = [idx for idx in knn_neighbors if idx != point_idx]
                manual_neighbors = manual_neighbors  # Keep manual unchanged
            else:
                # Add to manual list
                manual_neighbors.append(point_idx)
                new_knn = knn_neighbors  # Keep KNN unchanged

            # Keep the current display but return updated neighbor lists
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                manual_neighbors,
                dash.no_update,
                new_knn
            )

        @self.app.callback(
            Output("selected-image", "children", allow_duplicate=True),
            Output("selected-details", "children", allow_duplicate=True),
            Input("selected-point-store", "data"),
            prevent_initial_call=True
        )
        def update_selection_display(selected_idx):
            """Update selection display when selected point changes (e.g., after generation)."""
            if selected_idx is None or self.df.empty:
                return dash.no_update, dash.no_update

            # Get sample info
            sample = self.df.iloc[selected_idx]
            img_b64 = self.get_image_base64(sample['image_path'])
            img_element = html.Img(
                src=img_b64,
                style={"width": "100%", "border": "2px solid #0d6efd", "borderRadius": "4px"}
            ) if img_b64 else html.Div("Image not found")

            details = []
            details.append(html.P([html.Strong("Sample ID: "), sample['sample_id']]))
            class_name = self.get_sample_class_name(sample)
            if class_name is not None:
                class_id = int(sample['class_label'])
                details.append(html.P([html.Strong("Class: "), f"{class_id}: {class_name}"]))
            if 'conditioning_sigma' in sample and pd.notna(sample['conditioning_sigma']):
                details.append(html.P([html.Strong("Sigma: "), f"{sample['conditioning_sigma']}"]))
            details.append(html.P([
                html.Strong("UMAP Coords: "),
                f"({sample['umap_x']:.2f}, {sample['umap_y']:.2f})"
            ]))

            # Check if this is a generated sample
            is_generated_col = self.df.get('is_generated', pd.Series([False] * len(self.df)))
            if selected_idx < len(is_generated_col) and is_generated_col.iloc[selected_idx]:
                details.append(html.P(
                    "✓ Generated from neighbors",
                    className="text-success small font-weight-bold"
                ))
            else:
                details.append(html.P(
                    "Click points to add or remove neighbors",
                    className="text-info small"
                ))

            return img_element, html.Div(details)

        @self.app.callback(
            Output("neighbor-list", "children"),
            Input("manual-neighbors-store", "data"),
            Input("neighbor-indices-store", "data"),
            State("selected-point-store", "data"),
            prevent_initial_call=False
        )
        def display_neighbor_list(manual_neighbors, knn_neighbors, selected_idx):
            """Display combined neighbor list with remove buttons for manual neighbors."""
            if self.df.empty:
                return html.Div("No data loaded", className="text-muted")

            if not manual_neighbors and not knn_neighbors:
                return html.Div("No neighbors selected", className="text-muted small")

            # Ensure lists are not None
            manual_neighbors = manual_neighbors or []
            knn_neighbors = knn_neighbors or []

            # Filter out any indices that are now out of bounds (e.g., after clearing generated)
            max_idx = len(self.df) - 1
            manual_neighbors = [idx for idx in manual_neighbors if idx <= max_idx]
            knn_neighbors = [idx for idx in knn_neighbors if idx <= max_idx]

            # Build combined list (KNN first, then manual additions at bottom)
            all_neighbors = []
            # Add KNN neighbors that aren't in manual list
            for idx in knn_neighbors:
                if idx not in manual_neighbors:
                    all_neighbors.append(idx)
            # Add manual neighbors at the end (most recently added at bottom)
            all_neighbors.extend(manual_neighbors)

            if not all_neighbors:
                return html.Div("No neighbors selected", className="text-muted small")

            # Build neighbor cards
            neighbor_items = []
            for i, idx in enumerate(all_neighbors):
                neighbor_sample = self.df.iloc[idx]
                is_manual = idx in manual_neighbors

                # Get image thumbnail
                img_b64 = self.get_image_base64(neighbor_sample['image_path'], size=(64, 64))

                # Calculate distance if selected point exists and is valid
                dist_text = ""
                if selected_idx is not None and selected_idx <= max_idx:
                    selected_coords = self.df.iloc[selected_idx][['umap_x', 'umap_y']].values
                    neighbor_coords = self.df.iloc[idx][['umap_x', 'umap_y']].values
                    dist = np.linalg.norm(selected_coords - neighbor_coords)
                    dist_text = f"(dist: {dist:.2f})"

                # Build neighbor card
                neighbor_card = dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Img(
                                    src=img_b64,
                                    style={"width": "64px", "height": "64px"}
                                ) if img_b64 else html.Div("No img", style={"width": "64px"})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.Strong(f"#{i+1} "),
                                    html.Span("✓ " if is_manual else "", className="text-success"),
                                    html.Span(dist_text, className="text-muted small"),
                                    html.Br(),
                                    html.Span(
                                        f"{int(neighbor_sample['class_label'])}: {self.get_sample_class_name(neighbor_sample)}",
                                        className="small"
                                    ) if self.get_sample_class_name(neighbor_sample) is not None else None,
                                    html.Br() if 'conditioning_sigma' in neighbor_sample and pd.notna(neighbor_sample['conditioning_sigma']) else None,
                                    html.Span(
                                        f"σ={neighbor_sample['conditioning_sigma']}",
                                        className="small text-muted"
                                    ) if 'conditioning_sigma' in neighbor_sample and pd.notna(neighbor_sample['conditioning_sigma']) else None,
                                ])
                            ], width=8)
                        ])
                    ], className="p-2")
                ], className="mb-2", style={"border": "2px solid green" if is_manual else "1px solid #dee2e6"})

                neighbor_items.append(neighbor_card)

            return html.Div(neighbor_items)

        @self.app.callback(
            Output("umap-scatter", "figure", allow_duplicate=True),
            Input("selected-point-store", "data"),
            Input("manual-neighbors-store", "data"),
            Input("neighbor-indices-store", "data"),
            Input("highlighted-class-store", "data"),
            State("umap-scatter", "figure"),
            prevent_initial_call=True
        )
        def highlight_neighbors(selected_idx, manual_neighbors, neighbor_indices, highlighted_class, current_figure):
            """Highlight selected point, neighbors, and class filter on plot."""
            if current_figure is None:
                return current_figure

            fig = go.Figure(current_figure)

            # Remove any existing highlight traces but keep main scatter + generated overlay + trajectories
            # Trace 0 = main scatter, keep 'Generated', 'Trajectory', 'Intended' traces
            # Remove highlight traces (Selected, KNN Neighbors, Manual Neighbors, Class Highlight)
            base_traces = []
            for i, trace in enumerate(fig.data):
                trace_name = trace.name or ''
                # Keep main scatter, generated overlay, and trajectory traces
                if i == 0 or trace_name == 'Generated' or 'Trajectory' in trace_name or trace_name == 'Intended':
                    base_traces.append(trace)
            fig.data = base_traces

            # Add class highlight X markers if a class is selected
            self.add_class_highlight_trace(fig, highlighted_class)

            # If no point selected or index out of bounds, return with class highlights only
            if selected_idx is None or selected_idx >= len(self.df):
                return fig

            # Highlight selected point (green if generated, blue if original)
            selected_coords = self.df.iloc[[selected_idx]][['umap_x', 'umap_y']]
            is_generated_col = self.df.get('is_generated', pd.Series([False] * len(self.df)))
            is_selected_generated = is_generated_col.iloc[selected_idx] if selected_idx < len(is_generated_col) else False

            selection_color = '#00FF00' if is_selected_generated else 'blue'
            selection_name = 'Selected (Generated)' if is_selected_generated else 'Selected Point'

            fig.add_trace(go.Scatter(
                x=selected_coords['umap_x'],
                y=selected_coords['umap_y'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=selection_color,
                    symbol='circle-open',
                    line=dict(width=3, color=selection_color)
                ),
                name=selection_name,
                hoverinfo='skip',
                showlegend=True
            ))

            # Add KNN neighbors in red with thin line if any
            if neighbor_indices:
                knn_coords = self.df.iloc[neighbor_indices][['umap_x', 'umap_y']]

                fig.add_trace(go.Scatter(
                    x=knn_coords['umap_x'],
                    y=knn_coords['umap_y'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle-open',
                        line=dict(width=1, color='red')
                    ),
                    name='KNN Neighbors',
                    hoverinfo='skip',
                    showlegend=True
                ))

            # Add manual neighbors in red with thicker line if any
            if manual_neighbors:
                manual_coords = self.df.iloc[manual_neighbors][['umap_x', 'umap_y']]

                fig.add_trace(go.Scatter(
                    x=manual_coords['umap_x'],
                    y=manual_coords['umap_y'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle-open',
                        line=dict(width=2, color='red')
                    ),
                    name='Manual Neighbors',
                    hoverinfo='skip',
                    showlegend=True
                ))

            return fig

        @self.app.callback(
            Output("download-data", "data"),
            Input("export-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def export_data(n_clicks):
            """Export current embeddings to CSV."""
            if self.df.empty:
                return None

            return dcc.send_data_frame(
                self.df.to_csv,
                "dmd2_embeddings_export.csv",
                index=False
            )

        @self.app.callback(
            Output("class-filter-dropdown", "options"),
            Input("status-text", "children"),  # Triggered when plot updates
            prevent_initial_call=False
        )
        def populate_class_dropdown(status):
            """Populate class filter dropdown with available classes."""
            if self.df.empty or 'class_label' not in self.df.columns:
                return []

            # Get unique classes and sort by class ID
            unique_classes = self.df['class_label'].dropna().unique()
            unique_classes = sorted([int(c) for c in unique_classes])

            # Build options with class name
            options = []
            for class_id in unique_classes:
                class_name = self.get_class_name(class_id)
                options.append({
                    "label": f"{class_id}: {class_name}",
                    "value": class_id
                })

            return options

        @self.app.callback(
            Output("highlighted-class-store", "data"),
            Output("clear-class-highlight-btn", "disabled"),
            Output("class-filter-status", "children"),
            Output("class-filter-dropdown", "value"),
            Input("class-filter-dropdown", "value"),
            Input("clear-class-highlight-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def handle_class_filter(selected_class, clear_clicks):
            """Handle class selection and clear button."""
            ctx = callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Handle clear button
            if trigger_id == "clear-class-highlight-btn":
                return None, True, "", None

            # Handle class selection
            if selected_class is not None:
                # Count samples in this class
                if 'class_label' in self.df.columns:
                    count = (self.df['class_label'] == selected_class).sum()
                    status = f"{count} samples"
                else:
                    status = ""
                return selected_class, False, status, dash.no_update

            # Dropdown cleared
            return None, True, "", dash.no_update

        @self.app.callback(
            Output("generation-status", "children"),
            Output("umap-scatter", "figure", allow_duplicate=True),
            Output("selected-point-store", "data", allow_duplicate=True),
            Input("generate-from-neighbors-btn", "n_clicks"),
            State("manual-neighbors-store", "data"),
            State("neighbor-indices-store", "data"),
            State("selected-point-store", "data"),
            State("umap-scatter", "figure"),
            State("highlighted-class-store", "data"),
            State("num-steps-input", "value"),
            State("mask-steps-input", "value"),
            State("guidance-scale-input", "value"),
            State("sigma-max-input", "value"),
            State("sigma-min-input", "value"),
            prevent_initial_call=True
        )
        def generate_from_neighbors(n_clicks, manual_neighbors, knn_neighbors, selected_idx, current_figure, highlighted_class,
                                    num_steps, mask_steps, guidance_scale, sigma_max, sigma_min):
            """Generate new image from neighbor center activation."""
            try:
                # Validate inputs
                if selected_idx is None:
                    return "Error: No point selected", dash.no_update, dash.no_update

                # Combine all neighbors
                all_neighbors = []
                if manual_neighbors:
                    all_neighbors.extend(manual_neighbors)
                if knn_neighbors:
                    all_neighbors.extend([n for n in knn_neighbors if n not in all_neighbors])

                if not all_neighbors:
                    return "Error: No neighbors selected", dash.no_update, dash.no_update

                if self.activations is None:
                    return "Error: Activations not loaded", dash.no_update, dash.no_update

                # Validate/default generation parameters
                num_steps = num_steps if num_steps is not None else self.num_steps
                mask_steps = mask_steps if mask_steps is not None else self.mask_steps
                guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale
                sigma_max = sigma_max if sigma_max is not None else self.sigma_max
                sigma_min = sigma_min if sigma_min is not None else self.sigma_min

                # NEW: Average neighbors directly in high-D activation space (no inverse_transform!)
                print(f"[GEN] Neighbors: {all_neighbors}")
                neighbor_activations = self.activations[all_neighbors]
                center_activation = np.mean(neighbor_activations, axis=0, keepdims=True)
                print(f"[GEN] Averaged {len(all_neighbors)} neighbor activations in high-D space")
                print(f"[GEN] Center activation shape: {center_activation.shape}")

                # Forward transform to get intended 2D position for visualization
                center_scaled = center_activation
                if self.umap_scaler is not None:
                    center_scaled = self.umap_scaler.transform(center_activation)
                if self.umap_reducer is not None:
                    center_2d = self.umap_reducer.transform(center_scaled)
                    print(f"[GEN] Intended center in UMAP: ({center_2d[0,0]:.3f}, {center_2d[0,1]:.3f})")
                else:
                    # Fallback: average UMAP coords of neighbors
                    neighbor_coords = self.df.iloc[all_neighbors][['umap_x', 'umap_y']].values
                    center_2d = np.mean(neighbor_coords, axis=0).reshape(1, -1)
                    print(f"[GEN] Intended center (avg UMAP coords): ({center_2d[0,0]:.3f}, {center_2d[0,1]:.3f})")

                # Load adapter if not already loaded
                if self.adapter is None:
                    if self.checkpoint_path is None:
                        return "Error: No checkpoint path provided", dash.no_update, dash.no_update

                    print(f"Loading adapter '{self.adapter_name}' ({num_steps}-step)...")
                    AdapterClass = get_adapter(self.adapter_name)
                    self.adapter = AdapterClass.from_checkpoint(
                        self.checkpoint_path,
                        device=self.device,
                        label_dropout=self.label_dropout
                    )

                # Determine which layers were used (from UMAP params or default)
                layers = self.umap_params.get('layers', ['encoder_bottleneck', 'midblock'])
                if isinstance(layers, str):
                    layers = [layers]

                # Get layer shapes from adapter
                if not self.layer_shapes:
                    self.layer_shapes = self.adapter.get_layer_shapes()

                # Split center activation back into per-layer activations
                # This assumes layers are concatenated in sorted order (same as process_embeddings.py)
                activation_dict = {}
                offset = 0
                for layer_name in sorted(layers):
                    shape = self.layer_shapes[layer_name]
                    size = np.prod(shape)
                    layer_act_flat = center_activation[0, offset:offset+size]
                    offset += size

                    # Reshape to (1, C, H, W)
                    layer_act = unflatten_activation(
                        torch.from_numpy(layer_act_flat).float(),
                        shape
                    )
                    activation_dict[layer_name] = layer_act

                print(f"[GEN] Split activation into {len(activation_dict)} layers", flush=True)

                # Create activation masker using adapter interface
                print("[GEN] Creating activation masker...", flush=True)
                masker = ActivationMasker(self.adapter)
                for layer_name, activation in activation_dict.items():
                    print(f"[GEN] Setting mask for {layer_name}, shape={activation.shape}", flush=True)
                    masker.set_mask(layer_name, activation)

                # Register hooks
                print("[GEN] Registering hooks...", flush=True)
                masker.register_hooks(list(activation_dict.keys()))
                print(f"[GEN] Registered {len(masker._handles)} hooks", flush=True)

                print("[GEN] Starting image generation...", flush=True)

                # Generate image (use same class as selected point if available)
                # All data uses Standard ImageNet ordering - no remapping needed
                class_label = None
                if 'class_label' in self.df.columns:
                    class_label = int(self.df.iloc[selected_idx]['class_label'])
                    print(f"Using class_label: {class_label} ({self.get_class_name(class_label)})")

                # Use multi-step or single-step generation based on config
                if num_steps > 1:
                    mask_info = f", mask_steps={mask_steps or num_steps}"
                    print(f"Using {num_steps}-step generation{mask_info}")
                    # pylint: disable=unbalanced-tuple-unpacking
                    images, labels, trajectory_acts, intermediate_imgs = generate_with_mask_multistep(
                        self.adapter,
                        masker,
                        class_label=class_label,
                        num_steps=num_steps,
                        mask_steps=mask_steps,
                        sigma_max=sigma_max,
                        sigma_min=sigma_min,
                        guidance_scale=guidance_scale,
                        stochastic=True,
                        num_samples=1,
                        device=self.device,
                        extract_layers=sorted(layers),
                        return_trajectory=True,
                        return_intermediates=True
                    )
                else:
                    images, labels = generate_with_mask(
                        self.adapter,
                        masker,
                        class_label=class_label,
                        conditioning_sigma=sigma_max,
                        num_samples=1,
                        device=self.device
                    )
                    trajectory_acts = None  # No trajectory for single-step
                    intermediate_imgs = None  # No intermediates for single-step

                # Clean up masker hooks
                masker.remove_hooks()

                print("Image generated successfully")

                # Project trajectory through UMAP and save intermediate images
                trajectory_coords = []
                if trajectory_acts is not None and len(trajectory_acts) > 0:
                    # Create directory for intermediate images
                    intermediate_dir = self.data_dir / "images" / "intermediates"
                    intermediate_dir.mkdir(parents=True, exist_ok=True)

                    # Compute sigma schedule for this trajectory
                    sigmas = get_denoising_sigmas(
                        num_steps, sigma_max, sigma_min
                    ).cpu().numpy()

                    # Only do UMAP projection if reducer is available
                    if self.umap_reducer is not None:
                        print(f"[GEN] Projecting {len(trajectory_acts)} trajectory points through UMAP...")

                        for step_idx, act in enumerate(trajectory_acts):
                            # Apply scaler if used during UMAP training
                            act_scaled = act
                            if self.umap_scaler is not None:
                                act_scaled = self.umap_scaler.transform(act)
                            # Project to 2D
                            coords = self.umap_reducer.transform(act_scaled)

                            # Save intermediate image if available
                            img_path = None
                            if intermediate_imgs is not None and step_idx < len(intermediate_imgs):
                                img_filename = f"sample_{len(self.df):06d}_step{step_idx}.png"
                                img_path = f"images/intermediates/{img_filename}"
                                full_path = self.data_dir / img_path
                                Image.fromarray(intermediate_imgs[step_idx][0].numpy()).save(full_path)

                            # Get sigma for this step
                            step_sigma = float(sigmas[step_idx]) if step_idx < len(sigmas) else None

                            trajectory_coords.append({
                                'step': step_idx,
                                'sigma': step_sigma,
                                'x': float(coords[0, 0]),
                                'y': float(coords[0, 1]),
                                'image_path': img_path
                            })
                        print(f"[GEN] Trajectory: {[(t['step'], round(t['sigma'], 3), round(t['x'], 3), round(t['y'], 3)) for t in trajectory_coords]}")
                    else:
                        # No UMAP reducer - just save intermediate images without trajectory
                        print(f"[GEN] Saving {len(trajectory_acts)} intermediate images (no UMAP projection)")
                        for step_idx in range(len(trajectory_acts)):
                            if intermediate_imgs is not None and step_idx < len(intermediate_imgs):
                                img_filename = f"sample_{len(self.df):06d}_step{step_idx}.png"
                                img_path = f"images/intermediates/{img_filename}"
                                full_path = self.data_dir / img_path
                                Image.fromarray(intermediate_imgs[step_idx][0].numpy()).save(full_path)

                # Save the generated sample
                model_type = self.umap_params.get('model', 'imagenet')
                next_sample_id = f"sample_{len(self.df):06d}_generated"

                metadata = {
                    'sample_id': next_sample_id,
                    'class_label': int(labels[0]),
                    'model': model_type,
                    'generated_from_neighbors': all_neighbors,
                    'neighbor_center_umap': center_2d.tolist()[0],
                    'trajectory': trajectory_coords
                }

                sample_record = save_generated_sample(
                    images[0],
                    {},  # Activations captured via trajectory, not needed here
                    metadata,
                    self.data_dir,
                    next_sample_id
                )

                # Use final trajectory point as actual position, or center_2d if no trajectory
                if trajectory_coords:
                    final_x = trajectory_coords[-1]['x']
                    final_y = trajectory_coords[-1]['y']
                else:
                    final_x = center_2d[0, 0]
                    final_y = center_2d[0, 1]

                # Add to dataframe with actual UMAP coordinates
                new_row = pd.DataFrame([{
                    'sample_id': next_sample_id,
                    'image_path': sample_record['image_path'],
                    'class_label': int(labels[0]),
                    'umap_x': final_x,
                    'umap_y': final_y,
                    'is_generated': True
                }])

                # Store trajectory for visualization
                if not hasattr(self, 'generated_trajectories'):
                    self.generated_trajectories = []
                self.generated_trajectories.append({
                    'sample_id': next_sample_id,
                    'intended_x': float(center_2d[0, 0]),
                    'intended_y': float(center_2d[0, 1]),
                    'trajectory': trajectory_coords
                })
                # Mark existing points as not generated if column doesn't exist
                if 'is_generated' not in self.df.columns:
                    self.df['is_generated'] = False
                self.df = pd.concat([self.df, new_row], ignore_index=True)

                # Refit nearest neighbors
                self.fit_nearest_neighbors()

                # Regenerate entire plot with new point included
                new_idx = len(self.df) - 1

                # Determine color column
                if 'class_label' in self.df.columns:
                    color_col = 'class_label'
                    hover_data = ['class_label']
                else:
                    color_col = None
                    hover_data = []

                # Create figure with all points
                fig = px.scatter(
                    self.df,
                    x='umap_x',
                    y='umap_y',
                    color=color_col,
                    hover_data=hover_data + ['sample_id'],
                    title="DMD2 Activation UMAP",
                    labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2'}
                )

                fig.update_traces(marker=dict(size=5, opacity=0.7))

                # Add trajectory traces for generated samples
                if hasattr(self, 'generated_trajectories') and self.generated_trajectories:
                    for traj_data in self.generated_trajectories:
                        sample_id = traj_data['sample_id']
                        intended_x = traj_data['intended_x']
                        intended_y = traj_data['intended_y']
                        trajectory = traj_data.get('trajectory', [])

                        if trajectory:
                            # Build path: intended point -> step 0 -> step 1 -> ... -> final
                            path_x = [intended_x] + [t['x'] for t in trajectory]
                            path_y = [intended_y] + [t['y'] for t in trajectory]
                            # Include sigma in step labels if available
                            step_labels = []
                            for t in trajectory:
                                if t.get('sigma') is not None:
                                    step_labels.append(f"step {t['step']}, σ={t['sigma']:.3f}")
                                else:
                                    step_labels.append(f"step {t['step']}")
                            steps = ['intended'] + step_labels
                            # Image paths: None for intended, then step images
                            img_paths = [None] + [t.get('image_path') for t in trajectory]

                            # Add trajectory line (dotted)
                            fig.add_trace(go.Scatter(
                                x=path_x,
                                y=path_y,
                                mode='lines+markers',
                                line=dict(color='rgba(0, 180, 0, 0.7)', width=3, dash='dot'),
                                marker=dict(size=8, color='rgba(0, 180, 0, 0.8)'),
                                name=f'Trajectory: {sample_id}',
                                text=steps,
                                customdata=[[sample_id, step, img] for step, img in zip(steps, img_paths)],
                                hovertemplate=f'{sample_id}<br>%{{text}}<br>(%{{x:.3f}}, %{{y:.3f}})<extra></extra>',
                                showlegend=False
                            ))

                            # Add hollow circle at intended point
                            fig.add_trace(go.Scatter(
                                x=[intended_x],
                                y=[intended_y],
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color='rgba(255,255,255,0)',
                                    line=dict(width=2, color='#00CC00')
                                ),
                                name='Intended',
                                customdata=[[sample_id, 'intended', None]],  # Match trajectory format
                                hovertemplate=f'{sample_id}<br>intended<br>(%{{x:.3f}}, %{{y:.3f}})<extra></extra>',
                                showlegend=False
                            ))

                # Add bright overlay for generated points
                is_generated_col = self.df.get('is_generated', pd.Series([False] * len(self.df)))
                generated_df = self.df[is_generated_col]

                if len(generated_df) > 0:
                    # Get actual dataframe indices for generated samples
                    generated_indices = generated_df.index.tolist()

                    # Add bright green circles with black border as overlay
                    fig.add_trace(go.Scatter(
                        x=generated_df['umap_x'],
                        y=generated_df['umap_y'],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='#00FF00',  # Bright green
                            line=dict(width=2, color='#000000')  # Black border
                        ),
                        name='Generated',
                        text=generated_df['sample_id'],
                        customdata=[[idx] for idx in generated_indices],  # Store real df indices as list of lists
                        hovertemplate='<b>GENERATED: %{text}</b><extra></extra>',
                        showlegend=True
                    ))

                # Add class highlight if active
                self.add_class_highlight_trace(fig, highlighted_class)

                fig.update_layout(
                    hovermode='closest',
                    template='plotly_white'
                )

                success_msg = f"✓ Generated image saved as {next_sample_id}"
                return success_msg, fig, new_idx

            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                return error_msg, dash.no_update, dash.no_update

        # Clear Generated callback
        @self.app.callback(
            Output("umap-scatter", "figure", allow_duplicate=True),
            Output("generation-status", "children", allow_duplicate=True),
            Output("selected-point-store", "data", allow_duplicate=True),
            Output("manual-neighbors-store", "data", allow_duplicate=True),
            Output("neighbor-indices-store", "data", allow_duplicate=True),
            Output("selected-image", "children", allow_duplicate=True),
            Output("selected-details", "children", allow_duplicate=True),
            Input("clear-generated-btn", "n_clicks"),
            State("umap-scatter", "figure"),
            prevent_initial_call=True
        )
        def clear_generated(n_clicks, current_fig):
            if not n_clicks:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            # Remove generated samples from dataframe
            if 'is_generated' in self.df.columns:
                self.df = self.df[~self.df['is_generated']].reset_index(drop=True)

            # Clear stored trajectories
            if hasattr(self, 'generated_trajectories'):
                self.generated_trajectories = []

            # Refit nearest neighbors
            self.fit_nearest_neighbors()

            # Regenerate plot without generated points/trajectories
            color_col = 'class_label' if 'class_label' in self.df.columns else None
            hover_data = ['class_label'] if 'class_label' in self.df.columns else []

            fig = px.scatter(
                self.df,
                x='umap_x',
                y='umap_y',
                color=color_col,
                hover_data=hover_data + ['sample_id'],
                title="DMD2 Activation UMAP",
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2'}
            )
            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(hovermode='closest', template='plotly_white')

            # Reset selection state
            empty_selection_img = html.Div("Click a point to select", className="text-muted")
            empty_selection_details = html.Div("No sample selected", className="text-muted small")

            return (fig, "Generated samples cleared",
                    None, [], None,  # selected-point, manual-neighbors, knn-neighbors
                    empty_selection_img, empty_selection_details)

    def run(self, debug: bool = False, port: int = 8050):
        """Run the Dash app."""
        print(f"\nStarting DMD2 Visualizer on http://localhost:{port}")
        self.app.run(debug=debug, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="DMD2 Activation Visualizer"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to precomputed embeddings CSV (optional)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run server on"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to DMD2 checkpoint for generation (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device for generation"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1,
        help="Number of denoising steps (1=single-step, 4/10=multi-step)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="CFG scale (0=uncond, 1=class, >1=amplify, <0=anti-class)"
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum sigma for denoising schedule"
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum sigma for denoising schedule"
    )
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Label dropout (use 0.1 for CFG-trained models, including ~imagenet499.5k~ )"
    )
    parser.add_argument(
        "--mask_steps",
        type=int,
        default=None,
        help="Steps to apply activation mask (default=num_steps, use 1 for first-step-only)"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="dmd2-imagenet-64",
        help="Adapter name for model loading"
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--max_classes",
        "-c",
        type=int,
        default=None,
        help="Maximum classes to load (default: all)"
    )
    args = parser.parse_args()

    visualizer = DMD2Visualizer(
        data_dir=args.data_dir,
        embeddings_path=args.embeddings,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        num_steps=args.num_steps,
        mask_steps=args.mask_steps,
        guidance_scale=args.guidance_scale,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        label_dropout=args.label_dropout,
        adapter_name=args.adapter,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        max_classes=args.max_classes
    )

    visualizer.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()

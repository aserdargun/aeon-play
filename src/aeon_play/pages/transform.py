"""
Transformations Playground page for exploring aeon transformers.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..services.aeon_discovery import get_aeon_discovery
from ..services.runners import TaskRunner
from ..components.learning_section import create_learning_section
from ..components.level_gate import get_visible_items, get_level_features
from ..components.code_export import create_code_export_modal, create_export_button

dash.register_page(__name__, path="/transform", name="Transformations", title="aeon-play | Transformations")


# Curated transformer presets
TRANSFORMER_PRESETS = {
    "catch22": {
        "name": "Catch22 Features",
        "description": "Extract 22 canonical time series features",
        "transformer": "Catch22",
        "params": {},
        "level": "beginner",
    },
    "summary": {
        "name": "Summary Statistics",
        "description": "Basic statistical summaries",
        "transformer": "SummaryTransformer",
        "params": {},
        "level": "beginner",
    },
    "rocket": {
        "name": "ROCKET",
        "description": "Random convolutional kernel transform",
        "transformer": "Rocket",
        "params": {"num_kernels": 1000},
        "level": "intermediate",
    },
    "minirocket": {
        "name": "MiniRocket",
        "description": "Faster variant of ROCKET",
        "transformer": "MiniRocket",
        "params": {},
        "level": "intermediate",
    },
}


def create_transformer_selector(level: str):
    """Create transformer selection UI."""
    discovery = get_aeon_discovery()
    transformers = discovery.discover_estimators("transformer", level)

    options = [
        {"label": t.name, "value": t.name}
        for t in transformers[:30]  # Limit for UI
    ]

    return dbc.Card(
        [
            dbc.CardHeader("Select Transformer"),
            dbc.CardBody(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                [
                                    html.Div(
                                        [
                                            dbc.Button(
                                                p["name"],
                                                id={"type": "preset-btn", "index": key},
                                                color="outline-primary",
                                                size="sm",
                                                className="me-2 mb-2",
                                            )
                                            for key, p in TRANSFORMER_PRESETS.items()
                                            if get_level_features(level).get("all_estimators", False)
                                            or p["level"] in ["beginner", level]
                                        ],
                                        className="mb-2",
                                    ),
                                ],
                                label="Presets",
                            ),
                            dbc.Tab(
                                [
                                    dbc.Select(
                                        id="transform-selector",
                                        options=options,
                                        placeholder="Select a transformer...",
                                        className="mt-2",
                                    ),
                                ],
                                label="All Transformers",
                            ),
                        ],
                    ),
                ]
            ),
        ],
        className="mb-3",
    )


def create_param_controls():
    """Create parameter controls panel."""
    return dbc.Card(
        [
            dbc.CardHeader("Parameters"),
            dbc.CardBody(
                [
                    html.Div(id="transform-params-container"),
                    dbc.Button(
                        [html.I(className="fas fa-play me-2"), "Apply Transform"],
                        id="btn-apply-transform",
                        color="primary",
                        className="w-100 mt-3",
                    ),
                ]
            ),
        ],
    )


def create_before_after_plot(
    before: np.ndarray,
    after: np.ndarray,
    is_features: bool = False,
) -> go.Figure:
    """Create before/after transformation comparison plot."""
    if is_features:
        # Show feature values as bar chart
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Series", "Extracted Features"])

        fig.add_trace(
            go.Scatter(y=before.flatten()[:100], mode='lines', name='Series'),
            row=1, col=1
        )

        # Show first case's features
        features = after[0] if after.ndim > 1 else after
        fig.add_trace(
            go.Bar(y=features[:22], name='Features'),
            row=1, col=2
        )
    else:
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Before", "After"])

        fig.add_trace(
            go.Scatter(y=before.flatten()[:200], mode='lines', name='Before'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=after.flatten()[:200], mode='lines', name='After'),
            row=2, col=1
        )

    fig.update_layout(height=400, margin=dict(l=50, r=20, t=40, b=40))
    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Transformations Playground",
            content=(
                "Transformers convert time series into different representations. "
                "Feature extractors like Catch22 create tabular features, while "
                "others modify the series shape for downstream tasks."
            ),
            tips=[
                "Feature transformers output a 2D table (cases × features)",
                "Series transformers preserve the time series structure",
                "Transformers can be chained into pipelines",
            ],
        ),

        dbc.Row(
            [
                # Left panel
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data Source"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="transform-dataset-select",
                                            options=[],
                                            placeholder="Select dataset...",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Or use sample data", className="small"),
                                        dbc.Button(
                                            "Load Sample",
                                            id="btn-transform-sample",
                                            color="outline-secondary",
                                            size="sm",
                                            className="w-100",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(id="transform-selector-container"),
                        create_param_controls(),
                        html.Div(
                            [
                                create_export_button("btn-transform-export", "Generate Code"),
                            ],
                            className="mt-3",
                        ),
                    ],
                    md=4,
                ),

                # Right panel
                dbc.Col(
                    [
                        dcc.Loading(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Transformation Result"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(id="transform-plot"),
                                                html.Div(id="transform-output-info"),
                                            ]
                                        ),
                                    ],
                                ),
                                html.Div(id="transform-feature-table", className="mt-3"),
                            ],
                        ),
                    ],
                    md=8,
                ),
            ],
        ),

        create_code_export_modal("transform-code-modal"),
        dcc.Store(id="transform-current-X"),
        dcc.Store(id="transform-result"),
    ],
)


@callback(
    Output("transform-selector-container", "children"),
    Input("user-level", "data"),
)
def update_transformer_selector(level):
    return create_transformer_selector(level or "beginner")


@callback(
    Output("transform-dataset-select", "options"),
    Input("session-datasets", "data"),
)
def update_dataset_options(datasets):
    if not datasets:
        return []
    return [{"label": k, "value": k} for k in datasets.keys()]


@callback(
    Output("transform-current-X", "data"),
    Input("transform-dataset-select", "value"),
    Input("btn-transform-sample", "n_clicks"),
    State("session-datasets", "data"),
)
def load_transform_data(dataset_key, n_sample, datasets):
    triggered = ctx.triggered_id

    if triggered == "btn-transform-sample" or not dataset_key:
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(20, 1, 50)  # 20 cases, 1 channel, 50 timepoints
        return X.tolist()

    if datasets and dataset_key in datasets:
        return datasets[dataset_key]["X"]

    return None


@callback(
    Output("transform-params-container", "children"),
    Input("transform-selector", "value"),
    Input({"type": "preset-btn", "index": dash.ALL}, "n_clicks"),
    State("user-level", "data"),
)
def update_params(transformer, preset_clicks, level):
    triggered = ctx.triggered_id

    if isinstance(triggered, dict) and triggered.get("type") == "preset-btn":
        preset_key = triggered["index"]
        if preset_key in TRANSFORMER_PRESETS:
            preset = TRANSFORMER_PRESETS[preset_key]
            return html.Div(
                [
                    dbc.Alert(
                        [
                            html.Strong(preset["name"]),
                            html.Br(),
                            html.Small(preset["description"]),
                        ],
                        color="info",
                        className="mb-2",
                    ),
                    html.Small(f"Transformer: {preset['transformer']}", className="text-muted"),
                ]
            )

    if transformer:
        discovery = get_aeon_discovery()
        params = discovery.get_estimator_params(transformer, "transformer")

        if not params:
            return html.Small("No configurable parameters", className="text-muted")

        param_inputs = []
        for name, info in list(params.items())[:5]:  # Limit shown params
            param_inputs.append(
                html.Div(
                    [
                        dbc.Label(name.replace("_", " ").title(), className="small"),
                        dbc.Input(
                            id={"type": "transform-param", "name": name},
                            type="text",
                            value=str(info.get("default", "")),
                            size="sm",
                        ),
                    ],
                    className="mb-2",
                )
            )

        return html.Div(param_inputs)

    return html.Small("Select a transformer", className="text-muted")


@callback(
    Output("transform-plot", "figure"),
    Output("transform-output-info", "children"),
    Output("transform-feature-table", "children"),
    Input("btn-apply-transform", "n_clicks"),
    State("transform-current-X", "data"),
    State("transform-selector", "value"),
    prevent_initial_call=True,
)
def apply_transformation(n_clicks, X_data, transformer_name):
    if X_data is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data loaded")
        return empty_fig, "Load data first", None

    X = np.array(X_data)
    runner = TaskRunner()

    # Use default params for now
    success, result, error = runner.run_transformation(X, transformer_name or "Catch22", {})

    if not success:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {error}")
        return empty_fig, error, None

    # Determine if feature output
    is_features = result.ndim == 2 or (result.ndim == 3 and result.shape[1] != X.shape[1])

    fig = create_before_after_plot(X[0], result, is_features)

    # Output info
    info = html.Div(
        [
            html.P(f"Input shape: {X.shape}", className="mb-1 small"),
            html.P(f"Output shape: {result.shape}", className="mb-0 small"),
        ],
    )

    # Feature table for first few features
    table = None
    if is_features and result.ndim == 2:
        cols = min(10, result.shape[1])
        table = dbc.Table(
            [
                html.Thead(
                    html.Tr([html.Th(f"F{i}") for i in range(cols)])
                ),
                html.Tbody(
                    [
                        html.Tr([html.Td(f"{result[j, i]:.3f}") for i in range(cols)])
                        for j in range(min(3, result.shape[0]))
                    ]
                ),
            ],
            bordered=True,
            size="sm",
            className="mt-2",
        )

    return fig, info, table

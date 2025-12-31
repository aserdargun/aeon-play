"""
Segmentation task module for change point detection.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

from aeon_play.services.aeon_discovery import get_aeon_discovery
from aeon_play.services.runners import TaskRunner
from aeon_play.components.learning_section import create_learning_section
from aeon_play.components.metrics_display import create_run_logs

dash.register_page(
    __name__,
    path="/tasks/segmentation",
    name="Segmentation",
    title="aeon-play | Segmentation",
)


def create_segmentation_plot(
    series: np.ndarray,
    change_points: np.ndarray,
    true_cps: np.ndarray = None,
) -> go.Figure:
    """Create segmentation visualization."""
    fig = go.Figure()

    # Series
    fig.add_trace(
        go.Scatter(y=series, mode='lines', name='Series', line=dict(color='#2c3e50'))
    )

    # Detected change points
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for i, cp in enumerate(change_points):
        fig.add_vline(
            x=cp,
            line_dash="solid",
            line_color=colors[i % len(colors)],
            line_width=2,
            annotation_text=f"CP {i+1}",
            annotation_position="top",
        )

    # True change points if available
    if true_cps is not None:
        for cp in true_cps:
            fig.add_vline(
                x=cp,
                line_dash="dash",
                line_color="gray",
                line_width=1,
            )

    # Add segment shading
    all_cps = [0] + list(change_points) + [len(series)]
    segment_colors = ['rgba(52, 152, 219, 0.1)', 'rgba(46, 204, 113, 0.1)',
                      'rgba(155, 89, 182, 0.1)', 'rgba(241, 196, 15, 0.1)']

    for i in range(len(all_cps) - 1):
        fig.add_vrect(
            x0=all_cps[i],
            x1=all_cps[i + 1],
            fillcolor=segment_colors[i % len(segment_colors)],
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title="Segmentation Result",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Time Series Segmentation",
            content=(
                "Segmentation divides a time series into distinct segments by "
                "detecting change points where the statistical properties change. "
                "This is useful for activity recognition, regime detection, and more."
            ),
            tips=[
                "Change points mark transitions between different behaviors",
                "Number of expected segments can often be specified",
                "Consider both abrupt and gradual changes",
            ],
        ),

        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Generate Series"),
                                        dbc.Select(
                                            id="seg-series-type",
                                            options=[
                                                {"label": "Mean Shifts", "value": "mean_shifts"},
                                                {"label": "Variance Changes", "value": "variance"},
                                                {"label": "Trend Changes", "value": "trend"},
                                                {"label": "Frequency Changes", "value": "frequency"},
                                            ],
                                            value="mean_shifts",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Segmenter"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="seg-estimator-select",
                                            options=[],
                                            className="mb-2",
                                        ),
                                        dbc.Label("Expected Segments", className="small"),
                                        dbc.Input(
                                            id="seg-n-segments",
                                            type="number",
                                            min=2, max=10, value=3,
                                            size="sm",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Find Change Points"],
                            id="btn-seg-run",
                            color="primary",
                            className="w-100",
                        ),
                    ],
                    md=3,
                ),

                # Main
                dbc.Col(
                    [
                        dcc.Loading(
                            [
                                dbc.Alert(id="seg-status-alert", is_open=False, dismissable=True),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Segmentation Result"),
                                        dbc.CardBody(
                                            dcc.Graph(id="seg-result-plot", config={"displayModeBar": False})
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    md=6,
                ),

                # Results
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Results"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="seg-cp-list"),
                                        html.Div(id="seg-segment-info", className="mt-3"),
                                        html.Div(id="seg-logs-container", className="mt-3"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
        ),

        dcc.Store(id="seg-series-data"),
        dcc.Store(id="seg-true-cps"),
    ],
)


@callback(
    Output("seg-estimator-select", "options"),
    Input("user-level", "data"),
)
def update_segmenters(level):
    discovery = get_aeon_discovery()
    segmenters = discovery.discover_estimators("segmenter", level or "beginner")

    options = [{"label": s.name, "value": s.name} for s in segmenters]
    if not options:
        options = [{"label": "ClaSPSegmenter", "value": "ClaSPSegmenter"}]
    return options


@callback(
    Output("seg-series-data", "data"),
    Output("seg-true-cps", "data"),
    Input("seg-series-type", "value"),
)
def generate_series(series_type):
    np.random.seed(42)
    n = 300

    if series_type == "mean_shifts":
        series = np.concatenate([
            np.random.randn(100) + 0,
            np.random.randn(100) + 5,
            np.random.randn(100) + 2,
        ])
        true_cps = [100, 200]

    elif series_type == "variance":
        series = np.concatenate([
            np.random.randn(100) * 0.5,
            np.random.randn(100) * 2,
            np.random.randn(100) * 0.5,
        ])
        true_cps = [100, 200]

    elif series_type == "trend":
        t1 = np.linspace(0, 2, 100)
        t2 = np.linspace(0, 2, 100)
        t3 = np.linspace(0, 2, 100)
        series = np.concatenate([
            t1 * 2 + np.random.randn(100) * 0.3,
            -t2 * 3 + 4 + np.random.randn(100) * 0.3,
            t3 * 1 - 2 + np.random.randn(100) * 0.3,
        ])
        true_cps = [100, 200]

    else:  # frequency
        t = np.linspace(0, 2*np.pi, 100)
        series = np.concatenate([
            np.sin(2 * t) + 0.1 * np.random.randn(100),
            np.sin(6 * t) + 0.1 * np.random.randn(100),
            np.sin(2 * t) + 0.1 * np.random.randn(100),
        ])
        true_cps = [100, 200]

    return series.tolist(), true_cps


@callback(
    Output("seg-status-alert", "is_open"),
    Output("seg-status-alert", "children"),
    Output("seg-status-alert", "color"),
    Output("seg-result-plot", "figure"),
    Output("seg-cp-list", "children"),
    Output("seg-segment-info", "children"),
    Output("seg-logs-container", "children"),
    Input("btn-seg-run", "n_clicks"),
    State("seg-series-data", "data"),
    State("seg-true-cps", "data"),
    State("seg-estimator-select", "value"),
    State("seg-n-segments", "value"),
    prevent_initial_call=True,
)
def run_segmentation(n_clicks, series_data, true_cps_data, estimator, n_segments):
    if series_data is None:
        return True, "Generate series first", "warning", go.Figure(), None, None, None

    series = np.array(series_data)
    true_cps = np.array(true_cps_data) if true_cps_data else None

    # Simple change point detection for demo
    # Detect mean shifts using cumsum
    cumsum = np.cumsum(series - np.mean(series))
    diff = np.abs(np.diff(cumsum))

    # Find top n_segments-1 change points
    n_cps = (n_segments or 3) - 1
    cp_indices = np.argsort(diff)[-n_cps:]
    change_points = np.sort(cp_indices)

    fig = create_segmentation_plot(series, change_points, true_cps)

    # Change point list
    cp_list = html.Div(
        [
            html.H6("Detected Change Points"),
            html.Ul([html.Li(f"Position {cp}") for cp in change_points]),
        ]
    )

    # Segment info
    all_cps = [0] + list(change_points) + [len(series)]
    segment_info = html.Div(
        [
            html.H6("Segment Statistics"),
            html.Ul([
                html.Li(f"Segment {i+1}: [{all_cps[i]}, {all_cps[i+1]}] - "
                       f"mean={series[all_cps[i]:all_cps[i+1]].mean():.2f}")
                for i in range(len(all_cps) - 1)
            ]),
        ]
    )

    return (
        True, f"Found {len(change_points)} change points", "success",
        fig, cp_list, segment_info, None,
    )

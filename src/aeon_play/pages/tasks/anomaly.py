"""
Anomaly Detection task module.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeon_play.services.aeon_discovery import get_aeon_discovery
from aeon_play.services.runners import TaskRunner
from aeon_play.components.learning_section import create_learning_section
from aeon_play.components.metrics_display import create_metrics_display, create_run_logs

dash.register_page(
    __name__,
    path="/tasks/anomaly",
    name="Anomaly Detection",
    title="aeon-play | Anomaly Detection",
)


def create_anomaly_plot(
    series: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> go.Figure:
    """Create anomaly detection visualization."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Time Series with Anomalies", "Anomaly Scores"],
        row_heights=[0.6, 0.4],
    )

    # Series
    fig.add_trace(
        go.Scatter(y=series, mode='lines', name='Series', line=dict(color='#2c3e50')),
        row=1, col=1,
    )

    # Highlight anomalies
    if len(scores) == len(series):
        anomaly_mask = scores > threshold
        if np.any(anomaly_mask):
            anomaly_indices = np.where(anomaly_mask)[0]
            fig.add_trace(
                go.Scatter(
                    x=anomaly_indices,
                    y=series[anomaly_indices],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='#e74c3c', size=10),
                ),
                row=1, col=1,
            )

    # Scores
    fig.add_trace(
        go.Bar(y=scores, name='Scores', marker_color='#3498db'),
        row=2, col=1,
    )

    # Threshold
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(
        height=500,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Anomaly Detection",
            content=(
                "Anomaly detection identifies unusual patterns or outliers in time series. "
                "This is useful for fraud detection, system monitoring, and quality control."
            ),
            tips=[
                "Anomaly scores indicate how unusual each point is",
                "Adjust the threshold to balance precision and recall",
                "Consider domain knowledge when interpreting results",
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
                                            id="anom-series-type",
                                            options=[
                                                {"label": "Normal with Spikes", "value": "spikes"},
                                                {"label": "Seasonal with Outliers", "value": "seasonal"},
                                                {"label": "Level Shift", "value": "level_shift"},
                                                {"label": "Clean (No Anomalies)", "value": "clean"},
                                            ],
                                            value="spikes",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Detector"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="anom-estimator-select",
                                            options=[],
                                            className="mb-2",
                                        ),
                                        dbc.Label("Detection Threshold", className="small"),
                                        dcc.Slider(
                                            id="anom-threshold",
                                            min=0, max=1, step=0.05,
                                            value=0.5,
                                            marks={0: "0", 0.5: "0.5", 1: "1"},
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Detect Anomalies"],
                            id="btn-anom-run",
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
                                dbc.Alert(id="anom-status-alert", is_open=False, dismissable=True),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Detection Results"),
                                        dbc.CardBody(
                                            dcc.Graph(id="anom-result-plot", config={"displayModeBar": False})
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
                                        html.Div(id="anom-metrics-container"),
                                        html.Div(id="anom-summary", className="mt-3"),
                                        html.Div(id="anom-logs-container", className="mt-3"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
        ),

        dcc.Store(id="anom-series-data"),
        dcc.Store(id="anom-labels-data"),
    ],
)


@callback(
    Output("anom-estimator-select", "options"),
    Input("user-level", "data"),
)
def update_detectors(level):
    discovery = get_aeon_discovery()
    detectors = discovery.discover_estimators("anomaly_detector", level or "beginner")

    options = [{"label": d.name, "value": d.name} for d in detectors]
    if not options:
        options = [{"label": "STRAY", "value": "STRAY"}]
    return options


@callback(
    Output("anom-series-data", "data"),
    Output("anom-labels-data", "data"),
    Input("anom-series-type", "value"),
)
def generate_series(series_type):
    np.random.seed(42)
    n = 200

    series = np.zeros(n)
    labels = np.zeros(n)

    if series_type == "spikes":
        series = np.sin(np.linspace(0, 8*np.pi, n)) + 0.1 * np.random.randn(n)
        spike_indices = [30, 75, 120, 165]
        for idx in spike_indices:
            series[idx] += 3 * (1 if np.random.rand() > 0.5 else -1)
            labels[idx] = 1

    elif series_type == "seasonal":
        t = np.linspace(0, 4, n)
        series = 10 * np.sin(2 * np.pi * t) + 0.3 * np.random.randn(n)
        outlier_indices = [25, 90, 150]
        for idx in outlier_indices:
            series[idx] += 8
            labels[idx] = 1

    elif series_type == "level_shift":
        series = np.concatenate([
            np.random.randn(80),
            np.random.randn(40) + 5,
            np.random.randn(80),
        ])
        labels[78:82] = 1
        labels[118:122] = 1

    else:  # clean
        series = np.sin(np.linspace(0, 8*np.pi, n)) + 0.1 * np.random.randn(n)

    return series.tolist(), labels.tolist()


@callback(
    Output("anom-status-alert", "is_open"),
    Output("anom-status-alert", "children"),
    Output("anom-status-alert", "color"),
    Output("anom-result-plot", "figure"),
    Output("anom-metrics-container", "children"),
    Output("anom-summary", "children"),
    Output("anom-logs-container", "children"),
    Input("btn-anom-run", "n_clicks"),
    Input("anom-threshold", "value"),
    State("anom-series-data", "data"),
    State("anom-labels-data", "data"),
    State("anom-estimator-select", "value"),
    prevent_initial_call=True,
)
def run_detection(n_clicks, threshold, series_data, labels_data, estimator):
    triggered = ctx.triggered_id

    if series_data is None:
        return True, "Generate series first", "warning", go.Figure(), None, None, None

    series = np.array(series_data)
    labels = np.array(labels_data) if labels_data else None

    # Simple threshold-based scoring for demo
    # In real implementation, use the actual detector
    mean = np.mean(series)
    std = np.std(series)
    scores = np.abs(series - mean) / (std + 1e-10)
    scores = scores / scores.max()

    n_detected = np.sum(scores > threshold)

    fig = create_anomaly_plot(series, scores, threshold)

    metrics = html.Div(
        [
            html.P(f"Threshold: {threshold:.2f}", className="mb-1"),
            html.P(f"Detected: {n_detected} points", className="mb-0"),
        ]
    )

    summary = html.Div(
        [
            html.H6("Detection Summary"),
            html.P(f"Total points: {len(series)}"),
            html.P(f"Flagged as anomalies: {n_detected}"),
            html.P(f"Percentage: {100*n_detected/len(series):.1f}%"),
        ]
    )

    return (
        True if triggered == "btn-anom-run" else False,
        "Detection complete",
        "success",
        fig, metrics, summary, None,
    )

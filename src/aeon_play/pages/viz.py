"""
Visual Lab page for exploring time series visualizations.
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components.learning_section import create_learning_section
from ..components.level_gate import level_gate, get_level_features

dash.register_page(__name__, path="/viz", name="Visual Lab", title="aeon-play | Visual Lab")


def create_series_plot(series: np.ndarray, title: str = "Time Series") -> go.Figure:
    """Create a basic time series plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, mode='lines', name='Series'))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def create_lag_plot(series: np.ndarray, lag: int = 1) -> go.Figure:
    """Create a lag plot."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series[:-lag],
            y=series[lag:],
            mode='markers',
            marker=dict(size=5, opacity=0.5),
        )
    )
    fig.update_layout(
        title=f"Lag Plot (lag={lag})",
        xaxis_title=f"y(t)",
        yaxis_title=f"y(t+{lag})",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def create_acf_plot(series: np.ndarray, max_lag: int = 40) -> go.Figure:
    """Create an autocorrelation plot."""
    n = len(series)
    max_lag = min(max_lag, n - 1)
    mean = np.mean(series)
    var = np.var(series)

    acf = []
    for lag in range(max_lag + 1):
        if var > 0:
            c = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / (n * var)
        else:
            c = 1.0 if lag == 0 else 0.0
        acf.append(c)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(range(len(acf))), y=acf, marker_color='#3498db')
    )
    # Add confidence bounds
    conf = 1.96 / np.sqrt(n)
    fig.add_hline(y=conf, line_dash="dash", line_color="red")
    fig.add_hline(y=-conf, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Autocorrelation Function (ACF)",
        xaxis_title="Lag",
        yaxis_title="ACF",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def create_spectrogram(series: np.ndarray) -> go.Figure:
    """Create a simple power spectrum plot."""
    from scipy import signal

    # Compute periodogram
    freqs, psd = signal.periodogram(series)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=freqs, y=psd, mode='lines', fill='tozeroy')
    )
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency",
        yaxis_title="Power",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def create_stats_summary(series: np.ndarray) -> dbc.Card:
    """Create summary statistics card."""
    stats = {
        "Length": len(series),
        "Mean": f"{np.mean(series):.4f}",
        "Std": f"{np.std(series):.4f}",
        "Min": f"{np.min(series):.4f}",
        "Max": f"{np.max(series):.4f}",
        "Median": f"{np.median(series):.4f}",
    }

    rows = [
        html.Tr([html.Td(k, className="fw-bold"), html.Td(v)])
        for k, v in stats.items()
    ]

    return dbc.Card(
        [
            dbc.CardHeader("Summary Statistics"),
            dbc.CardBody(
                html.Table(
                    html.Tbody(rows),
                    className="table table-sm mb-0",
                )
            ),
        ],
    )


# Page layout
layout = html.Div(
    [
        create_learning_section(
            title="Visual Lab",
            content=(
                "Visualize and explore time series data. Understanding your data visually "
                "is crucial for selecting appropriate algorithms and interpreting results."
            ),
            tips=[
                "ACF plots help identify seasonality and correlation structure",
                "Lag plots can reveal non-linear patterns",
                "The power spectrum shows frequency components",
            ],
        ),

        dbc.Row(
            [
                # Controls
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data Source"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Use Dataset from Session"),
                                        dbc.Select(
                                            id="viz-dataset-select",
                                            options=[],
                                            placeholder="Select a dataset...",
                                            className="mb-2",
                                        ),
                                        html.Hr(),
                                        dbc.Label("Or Generate Sample Data"),
                                        dbc.Select(
                                            id="viz-sample-type",
                                            options=[
                                                {"label": "Sine Wave", "value": "sine"},
                                                {"label": "Random Walk", "value": "random_walk"},
                                                {"label": "Seasonal", "value": "seasonal"},
                                                {"label": "Trend + Noise", "value": "trend"},
                                                {"label": "AR(1) Process", "value": "ar1"},
                                            ],
                                            value="sine",
                                            className="mb-2",
                                        ),
                                        dbc.Button(
                                            "Generate",
                                            id="btn-generate-sample",
                                            color="primary",
                                            size="sm",
                                            className="w-100",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Series Selection"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Case Index", className="small"),
                                        dbc.Input(
                                            id="viz-case-idx",
                                            type="number",
                                            min=0,
                                            value=0,
                                            size="sm",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Channel Index", className="small"),
                                        dbc.Input(
                                            id="viz-channel-idx",
                                            type="number",
                                            min=0,
                                            value=0,
                                            size="sm",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Plot Options"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Lag (for lag plot)", className="small"),
                                        dbc.Input(
                                            id="viz-lag",
                                            type="number",
                                            min=1,
                                            max=50,
                                            value=1,
                                            size="sm",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Max ACF Lag", className="small"),
                                        dbc.Input(
                                            id="viz-max-acf-lag",
                                            type="number",
                                            min=10,
                                            max=100,
                                            value=40,
                                            size="sm",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),

                # Plots
                dbc.Col(
                    [
                        dcc.Loading(
                            id="loading-viz",
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Graph(id="viz-series-plot"),
                                            md=8,
                                        ),
                                        dbc.Col(
                                            html.Div(id="viz-stats-panel"),
                                            md=4,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Graph(id="viz-acf-plot"),
                                            md=6,
                                        ),
                                        dbc.Col(
                                            dcc.Graph(id="viz-lag-plot"),
                                            md=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(id="viz-advanced-plots"),
                            ],
                        ),
                    ],
                    md=9,
                ),
            ],
        ),

        # Store for current series
        dcc.Store(id="viz-current-series"),
    ],
)


@callback(
    Output("viz-dataset-select", "options"),
    Input("session-datasets", "data"),
)
def update_dataset_options(datasets):
    """Update dataset dropdown from session."""
    if not datasets:
        return []
    return [{"label": k, "value": k} for k in datasets.keys()]


@callback(
    Output("viz-current-series", "data"),
    Output("viz-series-plot", "figure"),
    Output("viz-stats-panel", "children"),
    Output("viz-acf-plot", "figure"),
    Output("viz-lag-plot", "figure"),
    Output("viz-advanced-plots", "children"),
    Input("btn-generate-sample", "n_clicks"),
    Input("viz-dataset-select", "value"),
    Input("viz-lag", "value"),
    Input("viz-max-acf-lag", "value"),
    Input("viz-case-idx", "value"),
    Input("viz-channel-idx", "value"),
    State("viz-sample-type", "value"),
    State("session-datasets", "data"),
    State("user-level", "data"),
)
def update_visualizations(
    n_clicks, dataset_key, lag, max_lag, case_idx, channel_idx,
    sample_type, datasets, level
):
    """Update all visualizations."""
    series = None

    # Get series from dataset or generate
    if dataset_key and datasets and dataset_key in datasets:
        X = np.array(datasets[dataset_key]["X"])
        if X.ndim == 3:
            case_idx = min(case_idx or 0, X.shape[0] - 1)
            channel_idx = min(channel_idx or 0, X.shape[1] - 1)
            series = X[case_idx, channel_idx, :]
        elif X.ndim == 2:
            case_idx = min(case_idx or 0, X.shape[0] - 1)
            series = X[case_idx, :]
        else:
            series = X.flatten()
    else:
        # Generate sample data
        np.random.seed(42)
        t = np.linspace(0, 10, 200)

        if sample_type == "sine":
            series = np.sin(2 * np.pi * t / 2) + 0.1 * np.random.randn(len(t))
        elif sample_type == "random_walk":
            series = np.cumsum(np.random.randn(len(t)))
        elif sample_type == "seasonal":
            series = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t) + 0.2 * np.random.randn(len(t))
        elif sample_type == "trend":
            series = 0.5 * t + np.random.randn(len(t))
        elif sample_type == "ar1":
            series = np.zeros(len(t))
            for i in range(1, len(t)):
                series[i] = 0.9 * series[i-1] + np.random.randn()
        else:
            series = np.random.randn(len(t))

    lag = lag or 1
    max_lag = max_lag or 40

    # Create plots
    series_fig = create_series_plot(series)
    stats_panel = create_stats_summary(series)
    acf_fig = create_acf_plot(series, max_lag)
    lag_fig = create_lag_plot(series, lag)

    # Advanced plots based on level
    features = get_level_features(level or "beginner")
    advanced = None

    if features.get("intermediate_viz", False):
        try:
            spectrum_fig = create_spectrogram(series)
            advanced = dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=spectrum_fig), md=12),
                ],
            )
        except Exception:
            advanced = None

    return (
        series.tolist(),
        series_fig,
        stats_panel,
        acf_fig,
        lag_fig,
        advanced,
    )

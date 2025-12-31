"""
Data preview component for displaying dataset information.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_data_preview(
    X: Any,
    y: Optional[Any],
    info: Dict,
    n_samples: int = 5,
    n_channels_show: int = 3,
) -> html.Div:
    """
    Create a comprehensive data preview component.

    Args:
        X: The data array
        y: Optional labels
        info: DatasetInfo dict
        n_samples: Number of sample series to show
        n_channels_show: Number of channels to show for multivariate

    Returns:
        Data preview component
    """
    # Statistics panel
    stats_panel = create_stats_panel(info)

    # Sample plot
    sample_plot = create_sample_plot(X, y, n_samples, n_channels_show)

    # Data contract
    contract_panel = create_data_contract_panel(info)

    # Class distribution if available
    class_dist = None
    if info.get("class_distribution"):
        class_dist = create_class_distribution(info["class_distribution"])

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(stats_panel, md=4),
                    dbc.Col(contract_panel, md=8),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(sample_plot, md=8),
                    dbc.Col(class_dist, md=4) if class_dist else None,
                ],
                className="mb-3",
            ),
        ],
        className="data-preview-container",
    )


def create_stats_panel(info: Dict) -> dbc.Card:
    """Create statistics panel."""
    stats = [
        ("Cases", info.get("n_cases", "N/A")),
        ("Channels", info.get("n_channels", 1)),
        ("Timepoints", info.get("n_timepoints", "N/A")),
        ("Memory", f"{info.get('memory_mb', 0):.2f} MB"),
    ]

    stat_elements = []
    for label, value in stats:
        stat_elements.append(
            dbc.Col(
                html.Div(
                    [
                        html.Div(str(value), className="value"),
                        html.Div(label, className="label"),
                    ],
                    className="data-stat",
                ),
                width=6,
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader("Dataset Statistics"),
            dbc.CardBody(dbc.Row(stat_elements)),
        ],
        className="h-100",
    )


def create_data_contract_panel(info: Dict) -> dbc.Card:
    """Create data contract panel showing compatibility."""
    badges = []

    # Data type badge
    data_type = info.get("data_type", "unknown")
    badges.append(
        dbc.Badge(
            data_type.replace("_", " ").title(),
            color="primary",
            className="me-1",
        )
    )

    # Univariate/Multivariate
    if info.get("is_univariate", True):
        badges.append(dbc.Badge("Univariate", color="success", className="me-1"))
    else:
        badges.append(dbc.Badge("Multivariate", color="info", className="me-1"))

    # Equal/Unequal length
    if info.get("is_equal_length", True):
        badges.append(dbc.Badge("Equal Length", color="success", className="me-1"))
    else:
        badges.append(dbc.Badge("Variable Length", color="warning", className="me-1"))

    # Has labels
    if info.get("has_labels"):
        badges.append(dbc.Badge("Labeled", color="success", className="me-1"))

    # Missing values
    if info.get("missing_values", 0) > 0:
        badges.append(
            dbc.Badge(
                f"{info['missing_values']} missing",
                color="warning",
                className="me-1",
            )
        )

    # Compatible tasks
    compatible_tasks = []
    if info.get("n_cases", 0) > 1:
        compatible_tasks.extend(["Classification", "Regression", "Clustering"])
    compatible_tasks.extend(["Forecasting", "Anomaly Detection", "Segmentation"])

    return dbc.Card(
        [
            dbc.CardHeader("Data Contract"),
            dbc.CardBody(
                [
                    html.Div(badges, className="mb-3"),
                    html.Hr(),
                    html.P(
                        [
                            html.Strong("Compatible Tasks: "),
                            ", ".join(compatible_tasks),
                        ],
                        className="mb-1 small",
                    ),
                    html.P(
                        [
                            html.Strong("Task Type: "),
                            info.get("task_type", "unknown").replace("_", " ").title(),
                        ],
                        className="mb-0 small",
                    ),
                ]
            ),
        ],
        className="h-100",
    )


def create_sample_plot(
    X: Any,
    y: Optional[Any],
    n_samples: int,
    n_channels: int,
) -> dcc.Graph:
    """Create sample series plot."""
    fig = go.Figure()

    # Handle different data formats
    if isinstance(X, np.ndarray):
        if X.ndim == 3:
            n_cases, n_ch, n_tp = X.shape
        elif X.ndim == 2:
            n_cases, n_tp = X.shape
            n_ch = 1
            X = X.reshape(n_cases, 1, n_tp)
        else:
            X = X.reshape(1, 1, -1)
            n_cases, n_ch, n_tp = 1, 1, len(X.flatten())

        # Select samples to plot
        sample_indices = np.linspace(0, n_cases - 1, min(n_samples, n_cases), dtype=int)
        channel_indices = range(min(n_channels, n_ch))

        colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

        for i, idx in enumerate(sample_indices):
            for ch in channel_indices:
                series = X[idx, ch, :]
                label_text = f"Case {idx}"
                if y is not None and len(y) > idx:
                    label_text += f" (y={y[idx]})"
                if n_ch > 1:
                    label_text += f" Ch{ch}"

                fig.add_trace(
                    go.Scatter(
                        y=series,
                        mode='lines',
                        name=label_text,
                        line=dict(
                            color=colors[i % len(colors)],
                            width=1.5,
                            dash='solid' if ch == 0 else 'dash',
                        ),
                        opacity=0.8,
                    )
                )

    elif isinstance(X, list):
        for i, arr in enumerate(X[:n_samples]):
            if arr.ndim > 1:
                arr = arr[0]  # First channel
            label_text = f"Case {i}"
            if y is not None and len(y) > i:
                label_text += f" (y={y[i]})"

            fig.add_trace(
                go.Scatter(
                    y=arr,
                    mode='lines',
                    name=label_text,
                    opacity=0.8,
                )
            )

    fig.update_layout(
        title="Sample Time Series",
        xaxis_title="Time",
        yaxis_title="Value",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_class_distribution(class_dist: Dict[str, int]) -> dbc.Card:
    """Create class distribution visualization."""
    labels = list(class_dist.keys())
    values = list(class_dist.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color='#3498db',
            )
        ]
    )

    fig.update_layout(
        title="Class Distribution",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Class",
        yaxis_title="Count",
    )

    return dbc.Card(
        [
            dbc.CardBody(
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ),
        ],
    )


def create_series_selector(
    n_cases: int,
    n_channels: int,
    prefix: str,
    selected_case: int = 0,
    selected_channel: int = 0,
) -> html.Div:
    """Create case/channel selector for single series operations."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Case Index", className="small"),
                            dbc.Input(
                                id=f"{prefix}-case-idx",
                                type="number",
                                min=0,
                                max=n_cases - 1,
                                value=selected_case,
                                size="sm",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Channel Index", className="small"),
                            dbc.Input(
                                id=f"{prefix}-channel-idx",
                                type="number",
                                min=0,
                                max=n_channels - 1,
                                value=selected_channel,
                                size="sm",
                            ),
                        ],
                        width=6,
                    ) if n_channels > 1 else None,
                ],
            ),
        ],
        className="mb-2",
    )


def create_data_warnings(info: Dict) -> Optional[dbc.Alert]:
    """Create warnings based on data characteristics."""
    warnings = []

    if info.get("n_cases", 0) > 5000:
        warnings.append("Large dataset - some operations may be slow")

    if isinstance(info.get("n_timepoints"), int) and info["n_timepoints"] > 2000:
        warnings.append("Long series - consider downsampling for faster exploration")

    if info.get("missing_values", 0) > 0:
        warnings.append(f"Dataset contains {info['missing_values']} missing values")

    if not info.get("is_equal_length", True):
        warnings.append("Unequal length series - some algorithms may not be compatible")

    if not warnings:
        return None

    return dbc.Alert(
        [
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Warnings: "),
            html.Ul([html.Li(w) for w in warnings], className="mb-0 mt-1"),
        ],
        color="warning",
        className="mb-3",
    )

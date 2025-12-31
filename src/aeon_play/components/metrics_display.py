"""
Metrics display component for showing model evaluation results.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff


def create_metrics_display(
    metrics: Dict[str, float],
    task_type: str,
) -> html.Div:
    """
    Create a metrics display panel.

    Args:
        metrics: Dict of metric_name -> value
        task_type: Type of task for formatting

    Returns:
        Metrics display component
    """
    metric_cards = []

    # Determine which metrics to highlight
    primary_metrics = {
        "classification": ["accuracy", "f1_weighted"],
        "regression": ["mae", "rmse", "r2"],
        "clustering": ["silhouette", "n_clusters"],
        "forecasting": ["mae", "rmse", "mape"],
        "anomaly_detection": ["precision", "recall", "n_anomalies"],
        "segmentation": ["n_change_points", "n_segments"],
    }

    highlight = primary_metrics.get(task_type, list(metrics.keys())[:3])

    for name, value in metrics.items():
        is_highlighted = name in highlight

        # Format value
        if isinstance(value, float):
            if name in ("accuracy", "f1_weighted", "precision", "recall", "r2", "silhouette"):
                formatted = f"{value:.1%}" if value <= 1 else f"{value:.4f}"
            else:
                formatted = f"{value:.4f}"
        else:
            formatted = str(value)

        card = html.Div(
            [
                html.Div(formatted, className="metric-value"),
                html.Div(
                    name.replace("_", " ").upper(),
                    className="metric-label",
                ),
            ],
            className=f"metric-card {'border-primary' if is_highlighted else ''}",
        )
        metric_cards.append(dbc.Col(card, xs=6, md=4, lg=3))

    return html.Div(
        [
            html.H6("Evaluation Metrics", className="mb-3"),
            dbc.Row(metric_cards),
        ],
    )


def create_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
) -> dcc.Graph:
    """
    Create a confusion matrix visualization.

    Args:
        cm: Confusion matrix array
        labels: Optional class labels

    Returns:
        Graph component with confusion matrix
    """
    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    # Normalize for color
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=300,
        margin=dict(l=60, r=20, t=40, b=60),
        yaxis=dict(autorange="reversed"),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dcc.Graph:
    """
    Create a residual plot for regression tasks.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Graph component with residual plot
    """
    residuals = y_true - y_pred

    fig = go.Figure()

    # Residual scatter
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='#3498db', size=8, opacity=0.6),
            name='Residuals',
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted",
        yaxis_title="Residual",
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dcc.Graph:
    """
    Create actual vs predicted scatter plot.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Graph component
    """
    fig = go.Figure()

    # Scatter
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker=dict(color='#3498db', size=8, opacity=0.6),
            name='Predictions',
        )
    )

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction',
        )
    )

    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_cluster_visualization(
    X: np.ndarray,
    labels: np.ndarray,
    n_samples_per_cluster: int = 3,
) -> dcc.Graph:
    """
    Create cluster visualization with sample series.

    Args:
        X: Data array
        labels: Cluster labels
        n_samples_per_cluster: Number of samples to show per cluster

    Returns:
        Graph component
    """
    from plotly.subplots import make_subplots

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig = make_subplots(
        rows=n_clusters,
        cols=1,
        subplot_titles=[f"Cluster {l}" for l in unique_labels],
        vertical_spacing=0.1,
    )

    colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        sample_indices = cluster_indices[:n_samples_per_cluster]

        for idx in sample_indices:
            if X.ndim == 3:
                series = X[idx, 0, :]
            else:
                series = X[idx]

            fig.add_trace(
                go.Scatter(
                    y=series,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=False,
                    opacity=0.7,
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=150 * n_clusters,
        margin=dict(l=50, r=20, t=30, b=20),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_forecast_plot(
    historical: np.ndarray,
    forecast: np.ndarray,
    actual: Optional[np.ndarray] = None,
    confidence_intervals: Optional[tuple] = None,
) -> dcc.Graph:
    """
    Create forecast visualization.

    Args:
        historical: Historical series
        forecast: Forecast values
        actual: Optional actual values for comparison
        confidence_intervals: Optional (lower, upper) bounds

    Returns:
        Graph component
    """
    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            y=historical,
            mode='lines',
            name='Historical',
            line=dict(color='#2c3e50', width=2),
        )
    )

    # Forecast
    forecast_x = list(range(len(historical), len(historical) + len(forecast)))
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#e74c3c', width=2, dash='dash'),
        )
    )

    # Actual if provided
    if actual is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color='#2ecc71', width=2),
            )
        )

    # Confidence intervals
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        fig.add_trace(
            go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=list(upper) + list(lower)[::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
            )
        )

    fig.update_layout(
        title="Forecast",
        xaxis_title="Time",
        yaxis_title="Value",
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_anomaly_plot(
    series: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dcc.Graph:
    """
    Create anomaly detection visualization.

    Args:
        series: Time series
        scores: Anomaly scores
        threshold: Detection threshold

    Returns:
        Graph component
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Time Series", "Anomaly Scores"],
        row_heights=[0.6, 0.4],
    )

    # Series with anomaly highlights
    anomaly_mask = scores > threshold if len(scores) == len(series) else np.zeros(len(series), dtype=bool)

    fig.add_trace(
        go.Scatter(
            y=series,
            mode='lines',
            name='Series',
            line=dict(color='#2c3e50', width=1.5),
        ),
        row=1,
        col=1,
    )

    # Highlight anomalies
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
            row=1,
            col=1,
        )

    # Scores
    fig.add_trace(
        go.Bar(
            y=scores,
            name='Scores',
            marker_color='#3498db',
        ),
        row=2,
        col=1,
    )

    # Threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_segmentation_plot(
    series: np.ndarray,
    change_points: np.ndarray,
) -> dcc.Graph:
    """
    Create segmentation visualization.

    Args:
        series: Time series
        change_points: Detected change point indices

    Returns:
        Graph component
    """
    fig = go.Figure()

    # Plot series
    fig.add_trace(
        go.Scatter(
            y=series,
            mode='lines',
            name='Series',
            line=dict(color='#2c3e50', width=1.5),
        )
    )

    # Add vertical lines for change points
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for i, cp in enumerate(change_points):
        fig.add_vline(
            x=cp,
            line_dash="dash",
            line_color=colors[i % len(colors)],
            annotation_text=f"CP {i+1}",
            annotation_position="top",
        )

    fig.update_layout(
        title="Segmentation Result",
        xaxis_title="Time",
        yaxis_title="Value",
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_run_logs(logs: List[str]) -> html.Div:
    """Create a log display component."""
    return html.Div(
        [
            html.H6("Run Log", className="mb-2"),
            html.Div(
                [html.Div(log, className="small text-muted") for log in logs],
                className="bg-light p-2 rounded",
                style={
                    "maxHeight": "200px",
                    "overflow": "auto",
                    "fontFamily": "monospace",
                    "fontSize": "0.8rem",
                },
            ),
        ],
    )

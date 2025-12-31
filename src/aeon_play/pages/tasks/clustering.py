"""
Clustering task module for time series clustering.
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
    path="/tasks/clustering",
    name="Clustering",
    title="aeon-play | Clustering",
)


def create_cluster_plot(X: np.ndarray, labels: np.ndarray) -> go.Figure:
    """Create cluster visualization with sample series per cluster."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig = make_subplots(
        rows=n_clusters,
        cols=1,
        subplot_titles=[f"Cluster {l} ({np.sum(labels == l)} series)" for l in unique_labels],
        vertical_spacing=0.1,
    )

    colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        sample_indices = cluster_indices[:5]

        for idx in sample_indices:
            series = X[idx, 0, :] if X.ndim == 3 else X[idx]
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

    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Time Series Clustering",
            content=(
                "Clustering groups similar time series together without labels. "
                "It's useful for discovering patterns, segmenting customers, or "
                "finding representative series (prototypes)."
            ),
            tips=[
                "TimeSeriesKMeans uses DTW barycenter averaging",
                "Silhouette score measures cluster quality",
                "If you have labels, ARI measures agreement with true clusters",
            ],
        ),

        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Dataset"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="clust-dataset-select",
                                            options=[],
                                            placeholder="Select dataset...",
                                        ),
                                        dbc.Button(
                                            "Load Sample Data",
                                            id="btn-clust-load-demo",
                                            color="outline-secondary",
                                            size="sm",
                                            className="w-100 mt-2",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Algorithm"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="clust-estimator-select",
                                            options=[],
                                            className="mb-2",
                                        ),
                                        dbc.Label("Number of Clusters", className="small"),
                                        dbc.Input(
                                            id="clust-n-clusters",
                                            type="number",
                                            min=2, max=20, value=3,
                                            size="sm",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Run Clustering"],
                            id="btn-clust-run",
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
                                dbc.Alert(id="clust-status-alert", is_open=False, dismissable=True),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Cluster Assignments"),
                                        dbc.CardBody(
                                            dcc.Graph(id="clust-result-plot", config={"displayModeBar": False})
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
                                        html.Div(id="clust-metrics-container"),
                                        html.Div(id="clust-distribution", className="mt-3"),
                                        html.Div(id="clust-logs-container", className="mt-3"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
        ),

        dcc.Store(id="clust-data-X"),
        dcc.Store(id="clust-data-y"),
    ],
)


@callback(
    Output("clust-estimator-select", "options"),
    Input("user-level", "data"),
)
def update_clusterers(level):
    discovery = get_aeon_discovery()
    clusterers = discovery.discover_estimators("clusterer", level or "beginner")
    return [{"label": c.name, "value": c.name} for c in clusterers]


@callback(
    Output("clust-dataset-select", "options"),
    Input("session-datasets", "data"),
)
def update_datasets(datasets):
    options = []
    if datasets:
        options.extend([{"label": k, "value": f"session:{k}"} for k in datasets.keys()])
    options.append({"label": "Sample Clustering Data", "value": "sample"})
    return options


@callback(
    Output("clust-data-X", "data"),
    Output("clust-data-y", "data"),
    Input("clust-dataset-select", "value"),
    Input("btn-clust-load-demo", "n_clicks"),
    State("session-datasets", "data"),
)
def load_data(dataset_key, n_demo, datasets):
    np.random.seed(42)

    if dataset_key and dataset_key.startswith("session:") and datasets:
        key = dataset_key.replace("session:", "")
        if key in datasets:
            X = np.array(datasets[key]["X"])
            y = datasets[key]["y"]
            return X.tolist(), y

    # Generate sample data with clear clusters
    n_per_cluster = 20
    n_timepoints = 50

    X = []
    y = []

    for c in range(3):
        for _ in range(n_per_cluster):
            if c == 0:
                series = np.sin(2 * np.pi * np.linspace(0, 2, n_timepoints)) + 0.2 * np.random.randn(n_timepoints)
            elif c == 1:
                series = np.cos(2 * np.pi * np.linspace(0, 2, n_timepoints)) + 0.2 * np.random.randn(n_timepoints)
            else:
                series = np.random.randn(n_timepoints) * 0.5
            X.append(series)
            y.append(c)

    X = np.array(X).reshape(-1, 1, n_timepoints)
    y = np.array(y)

    return X.tolist(), y.tolist()


@callback(
    Output("clust-status-alert", "is_open"),
    Output("clust-status-alert", "children"),
    Output("clust-status-alert", "color"),
    Output("clust-result-plot", "figure"),
    Output("clust-metrics-container", "children"),
    Output("clust-distribution", "children"),
    Output("clust-logs-container", "children"),
    Input("btn-clust-run", "n_clicks"),
    State("clust-data-X", "data"),
    State("clust-data-y", "data"),
    State("clust-estimator-select", "value"),
    State("clust-n-clusters", "value"),
    prevent_initial_call=True,
)
def run_clustering(n_clicks, X_data, y_data, estimator, n_clusters):
    if X_data is None:
        return True, "Load data first", "warning", go.Figure(), None, None, None

    X = np.array(X_data)
    y_true = np.array(y_data) if y_data else None

    runner = TaskRunner()
    result = runner.run_clustering(
        X=X,
        estimator_name=estimator or "TimeSeriesKMeans",
        params={"n_clusters": n_clusters or 3},
        y_true=y_true,
    )

    if result.success:
        labels = np.array(result.cluster_labels)
        fig = create_cluster_plot(X, labels)

        metrics = create_metrics_display(result.metrics, "clustering")

        # Cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = html.Div(
            [
                html.H6("Cluster Sizes", className="mb-2"),
                html.Ul([
                    html.Li(f"Cluster {u}: {c} series")
                    for u, c in zip(unique, counts)
                ]),
            ]
        )

        logs = create_run_logs(result.logs)

        return (
            True, f"Completed in {result.train_time:.2f}s", "success",
            fig, metrics, dist, logs,
        )
    else:
        return (
            True, result.error_message, "danger",
            go.Figure(), None, None, create_run_logs(result.logs) if result.logs else None,
        )

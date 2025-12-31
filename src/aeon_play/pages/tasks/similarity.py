"""
Similarity Search task module.
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

dash.register_page(
    __name__,
    path="/tasks/similarity",
    name="Similarity Search",
    title="aeon-play | Similarity Search",
)


def create_similarity_plot(
    query: np.ndarray,
    database: np.ndarray,
    top_k_indices: np.ndarray,
    distances: list = None,
) -> go.Figure:
    """Create similarity search visualization."""
    k = len(top_k_indices)

    fig = make_subplots(
        rows=k + 1,
        cols=1,
        subplot_titles=["Query"] + [f"Match {i+1} (dist={distances[i]:.3f})" if distances else f"Match {i+1}"
                                     for i in range(k)],
        vertical_spacing=0.05,
    )

    # Query
    fig.add_trace(
        go.Scatter(y=query, mode='lines', line=dict(color='#e74c3c', width=2), name='Query'),
        row=1, col=1,
    )

    # Matches
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    for i, idx in enumerate(top_k_indices):
        series = database[idx, 0, :] if database.ndim == 3 else database[idx]
        fig.add_trace(
            go.Scatter(
                y=series,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=1.5),
                name=f'Match {i+1}',
            ),
            row=i + 2, col=1,
        )

    fig.update_layout(
        height=120 * (k + 1),
        margin=dict(l=50, r=20, t=30, b=20),
        showlegend=False,
    )

    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Similarity Search",
            content=(
                "Similarity search finds time series in a database that are most similar "
                "to a query series. This is fundamental for pattern matching, "
                "retrieval systems, and nearest neighbor classification."
            ),
            tips=[
                "The choice of distance function greatly affects results",
                "k-NN search returns the k most similar series",
                "Consider normalization for fair comparisons",
            ],
        ),

        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Database"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="sim-dataset-select",
                                            options=[],
                                            placeholder="Select dataset...",
                                        ),
                                        dbc.Button(
                                            "Load Sample Database",
                                            id="btn-sim-load-demo",
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
                                dbc.CardHeader("Query"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Query Index", className="small"),
                                        dbc.Input(
                                            id="sim-query-idx",
                                            type="number",
                                            min=0, value=0,
                                            size="sm",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Top-K Results", className="small"),
                                        dbc.Input(
                                            id="sim-top-k",
                                            type="number",
                                            min=1, max=10, value=5,
                                            size="sm",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Distance"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="sim-distance-select",
                                            options=[
                                                {"label": "Euclidean", "value": "euclidean"},
                                                {"label": "DTW", "value": "dtw"},
                                                {"label": "Manhattan", "value": "manhattan"},
                                            ],
                                            value="euclidean",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-search me-2"), "Find Similar"],
                            id="btn-sim-run",
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
                                dbc.Alert(id="sim-status-alert", is_open=False, dismissable=True),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Search Results"),
                                        dbc.CardBody(
                                            dcc.Graph(id="sim-result-plot", config={"displayModeBar": False})
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
                                        html.Div(id="sim-results-list"),
                                        html.Div(id="sim-stats", className="mt-3"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
        ),

        dcc.Store(id="sim-database"),
    ],
)


@callback(
    Output("sim-dataset-select", "options"),
    Input("session-datasets", "data"),
)
def update_datasets(datasets):
    options = []
    if datasets:
        options.extend([{"label": k, "value": f"session:{k}"} for k in datasets.keys()])
    options.append({"label": "Sample Database", "value": "sample"})
    return options


@callback(
    Output("sim-database", "data"),
    Input("sim-dataset-select", "value"),
    Input("btn-sim-load-demo", "n_clicks"),
    State("session-datasets", "data"),
)
def load_database(dataset_key, n_demo, datasets):
    np.random.seed(42)

    if dataset_key and dataset_key.startswith("session:") and datasets:
        key = dataset_key.replace("session:", "")
        if key in datasets:
            return datasets[key]["X"]

    # Generate sample database with different patterns
    n_samples = 50
    n_timepoints = 100
    X = []

    patterns = [
        lambda t: np.sin(2 * np.pi * t / 20),
        lambda t: np.cos(2 * np.pi * t / 20),
        lambda t: np.sin(4 * np.pi * t / 20),
        lambda t: t / 100,
        lambda t: -t / 100 + 1,
    ]

    for i in range(n_samples):
        pattern_idx = i % len(patterns)
        t = np.arange(n_timepoints)
        series = patterns[pattern_idx](t) + 0.1 * np.random.randn(n_timepoints)
        X.append(series)

    X = np.array(X).reshape(n_samples, 1, n_timepoints)
    return X.tolist()


@callback(
    Output("sim-status-alert", "is_open"),
    Output("sim-status-alert", "children"),
    Output("sim-status-alert", "color"),
    Output("sim-result-plot", "figure"),
    Output("sim-results-list", "children"),
    Output("sim-stats", "children"),
    Input("btn-sim-run", "n_clicks"),
    State("sim-database", "data"),
    State("sim-query-idx", "value"),
    State("sim-top-k", "value"),
    State("sim-distance-select", "value"),
    prevent_initial_call=True,
)
def run_similarity_search(n_clicks, db_data, query_idx, top_k, distance):
    if db_data is None:
        return True, "Load database first", "warning", go.Figure(), None, None

    X = np.array(db_data)
    query_idx = min(query_idx or 0, len(X) - 1)
    top_k = top_k or 5

    query = X[query_idx, 0, :] if X.ndim == 3 else X[query_idx]

    # Compute distances
    X_flat = X.reshape(X.shape[0], -1)
    query_flat = query.flatten()

    if distance == "euclidean":
        distances = np.linalg.norm(X_flat - query_flat, axis=1)
    elif distance == "manhattan":
        distances = np.sum(np.abs(X_flat - query_flat), axis=1)
    else:  # dtw approximation
        distances = np.linalg.norm(X_flat - query_flat, axis=1)

    # Get top-k (excluding query itself)
    sorted_indices = np.argsort(distances)
    top_k_indices = sorted_indices[1:top_k + 1]  # Skip index 0 (query itself)
    top_k_distances = distances[top_k_indices]

    fig = create_similarity_plot(query, X, top_k_indices, top_k_distances.tolist())

    # Results list
    results_list = html.Div(
        [
            html.H6("Top-K Matches"),
            html.Ul([
                html.Li(f"Index {idx}: distance = {dist:.4f}")
                for idx, dist in zip(top_k_indices, top_k_distances)
            ]),
        ]
    )

    # Stats
    stats = html.Div(
        [
            html.H6("Search Statistics"),
            html.P(f"Database size: {len(X)}"),
            html.P(f"Query index: {query_idx}"),
            html.P(f"Distance: {distance}"),
            html.P(f"Min distance: {top_k_distances[0]:.4f}"),
            html.P(f"Max distance: {top_k_distances[-1]:.4f}"),
        ]
    )

    return (
        True, f"Found {top_k} similar series", "success",
        fig, results_list, stats,
    )

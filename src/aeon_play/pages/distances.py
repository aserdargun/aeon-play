"""
Distances Lab page for exploring time series distance functions.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..services.aeon_discovery import get_distance_functions
from ..components.learning_section import create_learning_section
from ..components.code_export import create_export_button

dash.register_page(__name__, path="/distances", name="Distances Lab", title="aeon-play | Distances Lab")


def compute_distance(series1: np.ndarray, series2: np.ndarray, method: str, params: dict) -> float:
    """Compute distance between two series."""
    try:
        if method == "euclidean":
            return float(np.sqrt(np.sum((series1 - series2) ** 2)))

        elif method == "squared":
            return float(np.sum((series1 - series2) ** 2))

        elif method == "manhattan":
            return float(np.sum(np.abs(series1 - series2)))

        elif method == "dtw":
            # Simple DTW implementation
            n, m = len(series1), len(series2)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = (series1[i-1] - series2[j-1]) ** 2
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],
                        dtw_matrix[i, j-1],
                        dtw_matrix[i-1, j-1]
                    )

            return float(np.sqrt(dtw_matrix[n, m]))

        else:
            # Fallback to Euclidean
            return float(np.sqrt(np.sum((series1 - series2) ** 2)))

    except Exception as e:
        return float('nan')


def compute_dtw_path(series1: np.ndarray, series2: np.ndarray) -> tuple:
    """Compute DTW and return the warping path."""
    n, m = len(series1), len(series2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (series1[i-1] - series2[j-1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            argmin = np.argmin([
                dtw_matrix[i-1, j-1],
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1]
            ])
            if argmin == 0:
                i, j = i-1, j-1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1

    path.reverse()
    return dtw_matrix[1:, 1:], path


def create_distance_comparison_plot(
    series1: np.ndarray,
    series2: np.ndarray,
    dtw_path: list = None,
) -> go.Figure:
    """Create distance comparison visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Series Comparison", "DTW Cost Matrix", "Alignment"],
        row_heights=[0.4, 0.6],
    )

    # Series comparison
    fig.add_trace(
        go.Scatter(y=series1, mode='lines', name='Series 1', line=dict(color='#2c3e50')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=series2, mode='lines', name='Series 2', line=dict(color='#e74c3c')),
        row=1, col=1
    )

    # DTW cost matrix and path
    if dtw_path:
        cost_matrix, path = dtw_path

        fig.add_trace(
            go.Heatmap(z=cost_matrix, colorscale='Blues', showscale=False),
            row=2, col=1
        )

        # Path on heatmap
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        fig.add_trace(
            go.Scatter(
                x=path_x, y=path_y,
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                name='Warping Path',
            ),
            row=2, col=1
        )

        # Alignment visualization
        for i, j in path[::max(1, len(path)//20)]:  # Sample path points
            fig.add_trace(
                go.Scatter(
                    x=[i, j + len(series1) + 10],
                    y=[series1[i], series2[j]],
                    mode='lines',
                    line=dict(color='rgba(52, 152, 219, 0.3)', width=1),
                    showlegend=False,
                ),
                row=2, col=2
            )

        fig.add_trace(
            go.Scatter(y=series1, mode='lines', name='Series 1', line=dict(color='#2c3e50')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(series2)) + len(series1) + 10,
                y=series2,
                mode='lines',
                name='Series 2',
                line=dict(color='#e74c3c'),
            ),
            row=2, col=2
        )

    fig.update_layout(height=600, margin=dict(l=50, r=20, t=40, b=40))
    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Distances Lab",
            content=(
                "Distance functions measure similarity between time series. "
                "Euclidean distance assumes alignment, while elastic measures like DTW "
                "allow for temporal warping. The choice of distance affects all downstream tasks."
            ),
            tips=[
                "Euclidean is fast but sensitive to shifts and stretching",
                "DTW aligns series optimally but is more expensive",
                "Many classifiers use distance-based nearest neighbor approaches",
            ],
        ),

        dbc.Row(
            [
                # Controls
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Series Selection"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Data Source"),
                                        dbc.Select(
                                            id="dist-dataset-select",
                                            options=[],
                                            placeholder="Use sample data",
                                            className="mb-2",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Series 1 Index", className="small"),
                                                        dbc.Input(
                                                            id="dist-series1-idx",
                                                            type="number",
                                                            min=0,
                                                            value=0,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Series 2 Index", className="small"),
                                                        dbc.Input(
                                                            id="dist-series2-idx",
                                                            type="number",
                                                            min=0,
                                                            value=1,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                        ),
                                        html.Hr(),
                                        dbc.Label("Sample Pattern"),
                                        dbc.Select(
                                            id="dist-sample-pattern",
                                            options=[
                                                {"label": "Sine vs Phase Shifted", "value": "phase"},
                                                {"label": "Sine vs Stretched", "value": "stretch"},
                                                {"label": "Sine vs Noisy", "value": "noise"},
                                                {"label": "Different Patterns", "value": "different"},
                                            ],
                                            value="phase",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Distance Function"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="dist-function-select",
                                            options=[],
                                            className="mb-2",
                                        ),
                                        html.Div(id="dist-params-container"),
                                        dbc.Button(
                                            [html.I(className="fas fa-calculator me-2"), "Compute Distance"],
                                            id="btn-compute-distance",
                                            color="primary",
                                            className="w-100 mt-3",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(id="dist-result-container", className="mt-3"),
                    ],
                    md=4,
                ),

                # Visualization
                dbc.Col(
                    [
                        dcc.Loading(
                            dcc.Graph(id="dist-comparison-plot"),
                        ),
                        html.Div(id="dist-explanation", className="mt-3"),
                    ],
                    md=8,
                ),
            ],
        ),

        dcc.Store(id="dist-series1"),
        dcc.Store(id="dist-series2"),
    ],
)


@callback(
    Output("dist-function-select", "options"),
    Input("user-level", "data"),
)
def update_distance_options(level):
    distances = get_distance_functions(level or "beginner")
    return [{"label": d["display_name"], "value": d["name"]} for d in distances]


@callback(
    Output("dist-dataset-select", "options"),
    Input("session-datasets", "data"),
)
def update_dataset_options(datasets):
    if not datasets:
        return []
    return [{"label": k, "value": k} for k in datasets.keys()]


@callback(
    Output("dist-series1", "data"),
    Output("dist-series2", "data"),
    Input("dist-dataset-select", "value"),
    Input("dist-sample-pattern", "value"),
    Input("dist-series1-idx", "value"),
    Input("dist-series2-idx", "value"),
    State("session-datasets", "data"),
)
def update_series(dataset_key, pattern, idx1, idx2, datasets):
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 100)

    if dataset_key and datasets and dataset_key in datasets:
        X = np.array(datasets[dataset_key]["X"])
        idx1 = min(idx1 or 0, len(X) - 1)
        idx2 = min(idx2 or 1, len(X) - 1)

        if X.ndim == 3:
            s1 = X[idx1, 0, :]
            s2 = X[idx2, 0, :]
        else:
            s1 = X[idx1]
            s2 = X[idx2]

        return s1.tolist(), s2.tolist()

    # Generate sample patterns
    if pattern == "phase":
        s1 = np.sin(t)
        s2 = np.sin(t + np.pi/4)  # Phase shifted
    elif pattern == "stretch":
        s1 = np.sin(t)
        s2 = np.sin(0.8 * t)  # Stretched
    elif pattern == "noise":
        s1 = np.sin(t)
        s2 = np.sin(t) + 0.3 * np.random.randn(len(t))
    else:
        s1 = np.sin(t)
        s2 = np.cos(t)

    return s1.tolist(), s2.tolist()


@callback(
    Output("dist-comparison-plot", "figure"),
    Output("dist-result-container", "children"),
    Output("dist-explanation", "children"),
    Input("btn-compute-distance", "n_clicks"),
    State("dist-series1", "data"),
    State("dist-series2", "data"),
    State("dist-function-select", "value"),
)
def compute_and_visualize(n_clicks, s1_data, s2_data, method):
    if s1_data is None or s2_data is None:
        return go.Figure(), None, None

    s1 = np.array(s1_data)
    s2 = np.array(s2_data)
    method = method or "euclidean"

    # Compute distance
    distance = compute_distance(s1, s2, method, {})

    # Compute DTW path for visualization
    dtw_path = None
    if method == "dtw":
        dtw_path = compute_dtw_path(s1, s2)

    fig = create_distance_comparison_plot(s1, s2, dtw_path)

    # Result display
    result = dbc.Card(
        dbc.CardBody(
            [
                html.H4(f"{distance:.4f}", className="text-primary text-center"),
                html.P(method.upper(), className="text-muted text-center mb-0"),
            ]
        ),
    )

    # Explanation
    explanations = {
        "euclidean": "Euclidean distance computes point-by-point differences. It assumes series are aligned in time.",
        "dtw": "Dynamic Time Warping finds the optimal alignment between series, allowing for stretching and compression.",
        "manhattan": "Manhattan distance sums absolute differences instead of squared differences.",
        "squared": "Squared Euclidean distance (no square root). Often used internally for efficiency.",
    }

    explanation = dbc.Alert(
        explanations.get(method, ""),
        color="info",
    )

    return fig, result, explanation

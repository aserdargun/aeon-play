"""
Benchmarking & Compare page for comparing multiple algorithms.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import time

from ..services.aeon_discovery import get_aeon_discovery
from ..services.datasets import get_dataset_service
from ..services.runners import TaskRunner
from ..components.learning_section import create_learning_section
from ..components.level_gate import get_level_features

dash.register_page(__name__, path="/benchmark", name="Benchmark", title="aeon-play | Benchmark")


def create_comparison_bar_chart(results: list, metric: str = "accuracy") -> go.Figure:
    """Create bar chart comparing algorithms."""
    names = [r["name"] for r in results]
    values = [r["metrics"].get(metric, 0) for r in results]
    times = [r["train_time"] for r in results]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=names,
            y=values,
            name=metric.upper(),
            marker_color='#3498db',
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
        )
    )

    fig.update_layout(
        title=f"Algorithm Comparison: {metric.upper()}",
        xaxis_title="Algorithm",
        yaxis_title=metric.upper(),
        height=400,
        margin=dict(l=50, r=20, t=40, b=100),
        xaxis_tickangle=-45,
    )

    return fig


def create_time_comparison(results: list) -> go.Figure:
    """Create training time comparison."""
    names = [r["name"] for r in results]
    times = [r["train_time"] for r in results]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=names,
            y=times,
            marker_color='#e74c3c',
            text=[f"{t:.2f}s" for t in times],
            textposition='auto',
        )
    )

    fig.update_layout(
        title="Training Time Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Time (seconds)",
        height=300,
        margin=dict(l=50, r=20, t=40, b=100),
        xaxis_tickangle=-45,
    )

    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Benchmarking & Comparison",
            content=(
                "Compare multiple algorithms on the same dataset to understand their "
                "trade-offs. This helps you choose the best approach for your specific problem."
            ),
            tips=[
                "Compare accuracy vs training time trade-offs",
                "Use cross-validation for robust comparisons",
                "Consider dataset characteristics when interpreting results",
            ],
        ),

        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Task Type"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="bench-task-type",
                                            options=[
                                                {"label": "Classification", "value": "classification"},
                                                {"label": "Regression", "value": "regression"},
                                                {"label": "Clustering", "value": "clustering"},
                                            ],
                                            value="classification",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Dataset"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="bench-dataset-select",
                                            options=[],
                                            placeholder="Select dataset...",
                                        ),
                                        dbc.Button(
                                            "Load GunPoint Demo",
                                            id="btn-bench-load-demo",
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
                                dbc.CardHeader("Algorithms"),
                                dbc.CardBody(
                                    [
                                        dbc.Checklist(
                                            id="bench-algorithm-checklist",
                                            options=[],
                                            value=[],
                                            className="mb-2",
                                        ),
                                        dbc.Button(
                                            "Select All",
                                            id="btn-bench-select-all",
                                            color="outline-secondary",
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
                                dbc.CardHeader("Settings"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Test Size", className="small"),
                                                        dbc.Input(
                                                            id="bench-test-size",
                                                            type="number",
                                                            min=0.1, max=0.5, step=0.05,
                                                            value=0.2, size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Random State", className="small"),
                                                        dbc.Input(
                                                            id="bench-random-state",
                                                            type="number",
                                                            value=42, size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Run Benchmark"],
                            id="btn-bench-run",
                            color="primary",
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), "Export Results"],
                            id="btn-bench-export",
                            color="outline-secondary",
                            className="w-100",
                        ),
                        dcc.Download(id="bench-download"),
                    ],
                    md=3,
                ),

                # Main content
                dbc.Col(
                    [
                        dcc.Loading(
                            [
                                dbc.Alert(id="bench-status-alert", is_open=False, dismissable=True),
                                dbc.Progress(id="bench-progress", className="mb-3", style={"display": "none"}),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Performance Comparison"),
                                        dbc.CardBody(
                                            dcc.Graph(id="bench-metric-plot", config={"displayModeBar": False})
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Training Time"),
                                        dbc.CardBody(
                                            dcc.Graph(id="bench-time-plot", config={"displayModeBar": False})
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            ],
                        ),
                    ],
                    md=6,
                ),

                # Results table
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Results Table"),
                                dbc.CardBody(
                                    html.Div(id="bench-results-table")
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
        ),

        dcc.Store(id="bench-data-X"),
        dcc.Store(id="bench-data-y"),
        dcc.Store(id="bench-results"),
    ],
)


@callback(
    Output("bench-algorithm-checklist", "options"),
    Output("bench-algorithm-checklist", "value"),
    Input("bench-task-type", "value"),
    Input("user-level", "data"),
)
def update_algorithms(task_type, level):
    discovery = get_aeon_discovery()
    level = level or "beginner"

    estimator_type = {
        "classification": "classifier",
        "regression": "regressor",
        "clustering": "clusterer",
    }.get(task_type, "classifier")

    estimators = discovery.discover_estimators(estimator_type, level)

    options = [{"label": e.name, "value": e.name} for e in estimators[:10]]
    default = [e.name for e in estimators[:3]]

    return options, default


@callback(
    Output("bench-algorithm-checklist", "value", allow_duplicate=True),
    Input("btn-bench-select-all", "n_clicks"),
    State("bench-algorithm-checklist", "options"),
    prevent_initial_call=True,
)
def select_all_algorithms(n_clicks, options):
    return [o["value"] for o in options]


@callback(
    Output("bench-dataset-select", "options"),
    Input("session-datasets", "data"),
    Input("bench-task-type", "value"),
)
def update_datasets(datasets, task_type):
    options = []
    if datasets:
        options.extend([{"label": k, "value": f"session:{k}"} for k in datasets.keys()])

    if task_type == "classification":
        options.extend([
            {"label": "GunPoint", "value": "builtin:GunPoint"},
            {"label": "ItalyPowerDemand", "value": "builtin:ItalyPowerDemand"},
        ])

    return options


@callback(
    Output("bench-data-X", "data"),
    Output("bench-data-y", "data"),
    Input("bench-dataset-select", "value"),
    Input("btn-bench-load-demo", "n_clicks"),
    State("session-datasets", "data"),
    State("bench-task-type", "value"),
)
def load_data(dataset_key, n_demo, datasets, task_type):
    triggered = ctx.triggered_id

    if triggered == "btn-bench-load-demo" or dataset_key is None:
        dataset_key = "builtin:GunPoint"

    if dataset_key and dataset_key.startswith("session:") and datasets:
        key = dataset_key.replace("session:", "")
        if key in datasets:
            return datasets[key]["X"], datasets[key]["y"]

    if dataset_key and dataset_key.startswith("builtin:"):
        name = dataset_key.replace("builtin:", "")
        service = get_dataset_service()
        try:
            X, y, info = service.load_builtin_dataset(name, "classification")
            return X.tolist(), y.tolist()
        except Exception:
            pass

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1, 50)
    y = np.array(["A"] * 50 + ["B"] * 50)
    return X.tolist(), y.tolist()


@callback(
    Output("bench-status-alert", "is_open"),
    Output("bench-status-alert", "children"),
    Output("bench-status-alert", "color"),
    Output("bench-metric-plot", "figure"),
    Output("bench-time-plot", "figure"),
    Output("bench-results-table", "children"),
    Output("bench-results", "data"),
    Input("btn-bench-run", "n_clicks"),
    State("bench-data-X", "data"),
    State("bench-data-y", "data"),
    State("bench-algorithm-checklist", "value"),
    State("bench-task-type", "value"),
    State("bench-test-size", "value"),
    State("bench-random-state", "value"),
    prevent_initial_call=True,
)
def run_benchmark(n_clicks, X_data, y_data, algorithms, task_type, test_size, random_state):
    if X_data is None or y_data is None:
        return True, "Load data first", "warning", go.Figure(), go.Figure(), None, None

    if not algorithms:
        return True, "Select at least one algorithm", "warning", go.Figure(), go.Figure(), None, None

    X = np.array(X_data)
    y = np.array(y_data)

    runner = TaskRunner()
    results = []

    for algo in algorithms:
        try:
            if task_type == "classification":
                result = runner.run_classification(
                    X=X, y=y,
                    estimator_name=algo,
                    params={},
                    test_size=test_size or 0.2,
                    random_state=random_state or 42,
                )
                if result.success:
                    results.append({
                        "name": algo,
                        "metrics": result.metrics,
                        "train_time": result.train_time,
                        "success": True,
                    })
                else:
                    results.append({
                        "name": algo,
                        "metrics": {"accuracy": 0},
                        "train_time": 0,
                        "success": False,
                        "error": result.error_message,
                    })

            elif task_type == "regression":
                result = runner.run_regression(
                    X=X, y=y.astype(float),
                    estimator_name=algo,
                    params={},
                    test_size=test_size or 0.2,
                    random_state=random_state or 42,
                )
                if result.success:
                    results.append({
                        "name": algo,
                        "metrics": result.metrics,
                        "train_time": result.train_time,
                        "success": True,
                    })

            elif task_type == "clustering":
                result = runner.run_clustering(
                    X=X,
                    estimator_name=algo,
                    params={"n_clusters": len(np.unique(y))},
                    y_true=y,
                )
                if result.success:
                    results.append({
                        "name": algo,
                        "metrics": result.metrics,
                        "train_time": result.train_time,
                        "success": True,
                    })

        except Exception as e:
            results.append({
                "name": algo,
                "metrics": {},
                "train_time": 0,
                "success": False,
                "error": str(e),
            })

    if not results:
        return True, "No results", "warning", go.Figure(), go.Figure(), None, None

    # Determine primary metric
    primary_metric = {
        "classification": "accuracy",
        "regression": "mae",
        "clustering": "silhouette",
    }.get(task_type, "accuracy")

    metric_fig = create_comparison_bar_chart(results, primary_metric)
    time_fig = create_time_comparison(results)

    # Results table
    table_header = [
        html.Thead(
            html.Tr([
                html.Th("Algorithm"),
                html.Th(primary_metric.upper()),
                html.Th("Time (s)"),
            ])
        )
    ]

    table_rows = []
    for r in sorted(results, key=lambda x: x["metrics"].get(primary_metric, 0), reverse=True):
        metric_val = r["metrics"].get(primary_metric, "N/A")
        if isinstance(metric_val, float):
            metric_val = f"{metric_val:.4f}"
        table_rows.append(
            html.Tr([
                html.Td(r["name"]),
                html.Td(metric_val),
                html.Td(f"{r['train_time']:.2f}"),
            ])
        )

    table = dbc.Table(
        table_header + [html.Tbody(table_rows)],
        bordered=True,
        striped=True,
        size="sm",
    )

    return (
        True, f"Benchmark complete: {len(results)} algorithms", "success",
        metric_fig, time_fig, table, results,
    )


@callback(
    Output("bench-download", "data"),
    Input("btn-bench-export", "n_clicks"),
    State("bench-results", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, results):
    if not results:
        return dash.no_update

    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    if results:
        metrics = list(results[0]["metrics"].keys())
        writer.writerow(["Algorithm"] + metrics + ["Train Time (s)", "Success"])

        for r in results:
            row = [r["name"]]
            row.extend([r["metrics"].get(m, "") for m in metrics])
            row.extend([r["train_time"], r["success"]])
            writer.writerow(row)

    return dcc.send_string(output.getvalue(), "benchmark_results.csv")

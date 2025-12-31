"""
Regression task module for time series regression.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

from aeon_play.services.aeon_discovery import get_aeon_discovery
from aeon_play.services.datasets import get_dataset_service
from aeon_play.services.runners import TaskRunner
from aeon_play.components.learning_section import create_learning_section
from aeon_play.components.metrics_display import (
    create_metrics_display,
    create_residual_plot,
    create_prediction_scatter,
    create_run_logs,
)

dash.register_page(
    __name__,
    path="/tasks/regression",
    name="Regression",
    title="aeon-play | Regression",
)


layout = html.Div(
    [
        create_learning_section(
            title="Time Series Regression",
            content=(
                "Regression predicts a continuous target value from time series. "
                "Unlike forecasting, regression predicts a single value (e.g., age, price) "
                "from the entire series, not future values."
            ),
            tips=[
                "Ensure your target variable is continuous, not categorical",
                "Consider feature extraction + standard regressor pipelines",
                "RMSE and MAE are common evaluation metrics",
            ],
        ),

        dbc.Row(
            [
                # Left sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Dataset"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="reg-dataset-select",
                                            options=[],
                                            placeholder="Select dataset...",
                                            className="mb-2",
                                        ),
                                        dbc.Button(
                                            "Load Sample Data",
                                            id="btn-reg-load-demo",
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
                                dbc.CardHeader("Algorithm"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="reg-estimator-select",
                                            options=[],
                                            className="mb-2",
                                        ),
                                        html.Div(id="reg-estimator-info", className="small text-muted"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Test Size", className="small"),
                                                        dbc.Input(
                                                            id="reg-test-size",
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
                                                            id="reg-random-state",
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
                            [html.I(className="fas fa-play me-2"), "Train & Evaluate"],
                            id="btn-reg-run",
                            color="primary",
                            className="w-100",
                        ),
                    ],
                    md=3,
                ),

                # Main content
                dbc.Col(
                    [
                        dcc.Loading(
                            [
                                dbc.Alert(id="reg-status-alert", is_open=False, dismissable=True),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Data Overview"),
                                        dbc.CardBody(
                                            [
                                                html.Div(id="reg-data-overview"),
                                                dcc.Graph(id="reg-target-dist-plot", config={"displayModeBar": False}),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Predictions"),
                                        dbc.CardBody(
                                            dbc.Row(
                                                [
                                                    dbc.Col(html.Div(id="reg-scatter-container"), md=6),
                                                    dbc.Col(html.Div(id="reg-residual-container"), md=6),
                                                ]
                                            )
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
                                        html.Div(id="reg-metrics-container"),
                                        html.Div(id="reg-logs-container", className="mt-3"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
        ),

        dcc.Store(id="reg-data-X"),
        dcc.Store(id="reg-data-y"),
    ],
)


@callback(
    Output("reg-estimator-select", "options"),
    Input("user-level", "data"),
)
def update_regressors(level):
    discovery = get_aeon_discovery()
    regressors = discovery.discover_estimators("regressor", level or "beginner")
    return [{"label": r.name, "value": r.name} for r in regressors]


@callback(
    Output("reg-dataset-select", "options"),
    Input("session-datasets", "data"),
)
def update_datasets(datasets):
    options = []
    if datasets:
        options.extend([{"label": k, "value": f"session:{k}"} for k in datasets.keys()])
    options.append({"label": "Sample Regression Data", "value": "sample"})
    return options


@callback(
    Output("reg-data-X", "data"),
    Output("reg-data-y", "data"),
    Output("reg-data-overview", "children"),
    Output("reg-target-dist-plot", "figure"),
    Input("reg-dataset-select", "value"),
    Input("btn-reg-load-demo", "n_clicks"),
    State("session-datasets", "data"),
)
def load_regression_data(dataset_key, n_demo, datasets):
    # Generate sample regression data
    np.random.seed(42)
    n_samples = 100
    n_timepoints = 50

    X = np.zeros((n_samples, 1, n_timepoints))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        freq = 0.5 + 0.1 * i / n_samples
        X[i, 0, :] = np.sin(2 * np.pi * freq * np.linspace(0, 1, n_timepoints))
        y[i] = freq * 10 + np.random.randn() * 0.5

    if dataset_key and dataset_key.startswith("session:") and datasets:
        key = dataset_key.replace("session:", "")
        if key in datasets:
            X = np.array(datasets[key]["X"])
            y = np.array(datasets[key]["y"]) if datasets[key]["y"] else y

    overview = html.Div(
        [
            html.P(f"Cases: {X.shape[0]}", className="mb-1"),
            html.P(f"Timepoints: {X.shape[-1]}", className="mb-1"),
            html.P(f"Target range: [{y.min():.2f}, {y.max():.2f}]", className="mb-0"),
        ]
    )

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y, nbinsx=20))
    fig.update_layout(
        title="Target Distribution",
        height=200,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return X.tolist(), y.tolist(), overview, fig


@callback(
    Output("reg-status-alert", "is_open"),
    Output("reg-status-alert", "children"),
    Output("reg-status-alert", "color"),
    Output("reg-metrics-container", "children"),
    Output("reg-scatter-container", "children"),
    Output("reg-residual-container", "children"),
    Output("reg-logs-container", "children"),
    Input("btn-reg-run", "n_clicks"),
    State("reg-data-X", "data"),
    State("reg-data-y", "data"),
    State("reg-estimator-select", "value"),
    State("reg-test-size", "value"),
    State("reg-random-state", "value"),
    prevent_initial_call=True,
)
def run_regression(n_clicks, X_data, y_data, estimator, test_size, random_state):
    if X_data is None or y_data is None:
        return True, "Load data first", "warning", None, None, None, None

    X = np.array(X_data)
    y = np.array(y_data)

    runner = TaskRunner()
    result = runner.run_regression(
        X=X, y=y,
        estimator_name=estimator or "KNeighborsTimeSeriesRegressor",
        params={},
        test_size=test_size or 0.2,
        random_state=random_state or 42,
    )

    if result.success:
        metrics = create_metrics_display(result.metrics, "regression")

        # For scatter and residual plots, we need actual vs predicted
        # Using result.predictions for now
        predictions = np.array(result.predictions) if result.predictions is not None else None

        scatter = None
        residual = None
        if predictions is not None:
            # Approximate test set
            n_test = len(predictions)
            y_test = y[-n_test:]
            scatter = create_prediction_scatter(y_test, predictions)
            residual = create_residual_plot(y_test, predictions)

        logs = create_run_logs(result.logs)

        return (
            True, f"Completed in {result.train_time:.2f}s", "success",
            metrics, scatter, residual, logs,
        )
    else:
        return (
            True, result.error_message, "danger",
            None, None, None, create_run_logs(result.logs) if result.logs else None,
        )

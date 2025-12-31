"""
Classification task module for time series classification.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

from ...services.aeon_discovery import get_aeon_discovery
from ...services.datasets import get_dataset_service
from ...services.runners import TaskRunner
from ...services.exporters import CodeExporter
from ...components.learning_section import create_learning_section
from ...components.level_gate import get_level_features
from ...components.metrics_display import (
    create_metrics_display,
    create_confusion_matrix,
    create_run_logs,
)
from ...components.code_export import create_code_export_modal

dash.register_page(
    __name__,
    path="/tasks/classification",
    name="Classification",
    title="aeon-play | Classification",
)


def create_classifier_selector(level: str):
    """Create classifier selection dropdown."""
    discovery = get_aeon_discovery()
    classifiers = discovery.discover_estimators("classifier", level)

    options = [{"label": c.name, "value": c.name} for c in classifiers]

    return dbc.Select(
        id="clf-estimator-select",
        options=options,
        value=options[0]["value"] if options else None,
        className="mb-2",
    )


layout = html.Div(
    [
        create_learning_section(
            title="Time Series Classification",
            content=(
                "Classification assigns labels to time series based on their patterns. "
                "aeon provides state-of-the-art classifiers including distance-based, "
                "feature-based, and ensemble methods."
            ),
            tips=[
                "Start with KNeighborsTimeSeriesClassifier for interpretable results",
                "ROCKET and Arsenal are fast and accurate for many datasets",
                "Consider the trade-off between accuracy and training time",
            ],
        ),

        dbc.Row(
            [
                # Left sidebar
                dbc.Col(
                    [
                        # Dataset selection
                        dbc.Card(
                            [
                                dbc.CardHeader("Dataset"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Select Dataset"),
                                        dbc.Select(
                                            id="clf-dataset-select",
                                            options=[],
                                            placeholder="Load from session or built-in...",
                                            className="mb-2",
                                        ),
                                        dbc.Button(
                                            "Load GunPoint Demo",
                                            id="btn-clf-load-demo",
                                            color="outline-secondary",
                                            size="sm",
                                            className="w-100",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),

                        # Algorithm selection
                        dbc.Card(
                            [
                                dbc.CardHeader("Algorithm"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="clf-estimator-container"),
                                        html.Div(id="clf-estimator-info", className="small text-muted mt-2"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),

                        # Parameters
                        dbc.Card(
                            [
                                dbc.CardHeader("Parameters"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="clf-params-container"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Test Size", className="small"),
                                                        dbc.Input(
                                                            id="clf-test-size",
                                                            type="number",
                                                            min=0.1,
                                                            max=0.5,
                                                            step=0.05,
                                                            value=0.2,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Random State", className="small"),
                                                        dbc.Input(
                                                            id="clf-random-state",
                                                            type="number",
                                                            value=42,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="clf-cv-container"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),

                        # Run button
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Train & Evaluate"],
                            id="btn-clf-run",
                            color="primary",
                            className="w-100 mb-3",
                        ),

                        dbc.Button(
                            [html.I(className="fas fa-code me-2"), "Generate Code"],
                            id="btn-clf-export",
                            color="outline-secondary",
                            className="w-100",
                        ),
                    ],
                    md=3,
                    className="task-sidebar",
                ),

                # Main content
                dbc.Col(
                    [
                        dcc.Loading(
                            [
                                dbc.Alert(
                                    id="clf-status-alert",
                                    is_open=False,
                                    dismissable=True,
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("Data Overview"),
                                                    dbc.CardBody(id="clf-data-overview"),
                                                ],
                                            ),
                                            md=6,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("Class Distribution"),
                                                    dbc.CardBody(
                                                        dcc.Graph(
                                                            id="clf-class-dist-plot",
                                                            config={"displayModeBar": False},
                                                        )
                                                    ),
                                                ],
                                            ),
                                            md=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Sample Series by Class"),
                                        dbc.CardBody(
                                            dcc.Graph(
                                                id="clf-sample-plot",
                                                config={"displayModeBar": False},
                                            )
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            ],
                        ),
                    ],
                    md=6,
                    className="task-main",
                ),

                # Results sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Results"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="clf-metrics-container"),
                                        html.Div(id="clf-confusion-container", className="mt-3"),
                                        html.Div(id="clf-logs-container", className="mt-3"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                    className="task-results",
                ),
            ],
        ),

        create_code_export_modal("clf-code-modal"),

        # Stores
        dcc.Store(id="clf-data-X"),
        dcc.Store(id="clf-data-y"),
        dcc.Store(id="clf-result"),
    ],
)


@callback(
    Output("clf-estimator-container", "children"),
    Output("clf-cv-container", "children"),
    Input("user-level", "data"),
)
def update_estimator_options(level):
    level = level or "beginner"
    features = get_level_features(level)

    selector = create_classifier_selector(level)

    cv_options = None
    if features.get("cross_validation", False):
        cv_options = html.Div(
            [
                dbc.Checkbox(
                    id="clf-use-cv",
                    label="Use Cross-Validation",
                    value=False,
                    className="mb-2",
                ),
                dbc.Input(
                    id="clf-cv-folds",
                    type="number",
                    min=2,
                    max=10,
                    value=5,
                    size="sm",
                    placeholder="CV Folds",
                ),
            ],
        )

    return selector, cv_options


@callback(
    Output("clf-dataset-select", "options"),
    Input("session-datasets", "data"),
)
def update_dataset_options(datasets):
    options = []
    if datasets:
        options.extend([{"label": k, "value": f"session:{k}"} for k in datasets.keys()])

    # Add built-in options
    options.extend([
        {"label": "GunPoint (Built-in)", "value": "builtin:GunPoint"},
        {"label": "ItalyPowerDemand (Built-in)", "value": "builtin:ItalyPowerDemand"},
        {"label": "ECG200 (Built-in)", "value": "builtin:ECG200"},
    ])

    return options


@callback(
    Output("clf-data-X", "data"),
    Output("clf-data-y", "data"),
    Output("clf-data-overview", "children"),
    Output("clf-class-dist-plot", "figure"),
    Output("clf-sample-plot", "figure"),
    Input("clf-dataset-select", "value"),
    Input("btn-clf-load-demo", "n_clicks"),
    State("session-datasets", "data"),
)
def load_classification_data(dataset_key, n_demo, datasets):
    triggered = ctx.triggered_id

    X, y = None, None

    if triggered == "btn-clf-load-demo" or dataset_key is None:
        dataset_key = "builtin:GunPoint"

    if dataset_key:
        if dataset_key.startswith("session:"):
            key = dataset_key.replace("session:", "")
            if datasets and key in datasets:
                X = np.array(datasets[key]["X"])
                y = np.array(datasets[key]["y"]) if datasets[key]["y"] else None
        elif dataset_key.startswith("builtin:"):
            name = dataset_key.replace("builtin:", "")
            service = get_dataset_service()
            try:
                X, y, info = service.load_builtin_dataset(name, "classification")
            except Exception as e:
                pass

    if X is None:
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 1, 50)
        y = np.array(["A"] * 50 + ["B"] * 50)

    # Data overview
    overview = html.Div(
        [
            html.P(f"Cases: {X.shape[0]}", className="mb-1"),
            html.P(f"Channels: {X.shape[1] if X.ndim > 2 else 1}", className="mb-1"),
            html.P(f"Timepoints: {X.shape[-1]}", className="mb-1"),
            html.P(f"Classes: {len(np.unique(y)) if y is not None else 'N/A'}", className="mb-0"),
        ]
    )

    # Class distribution plot
    class_fig = go.Figure()
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        class_fig.add_trace(go.Bar(x=unique.astype(str), y=counts))
    class_fig.update_layout(height=200, margin=dict(l=40, r=20, t=20, b=40))

    # Sample series plot
    sample_fig = go.Figure()
    if y is not None:
        unique_classes = np.unique(y)[:3]
        colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71']
        for i, cls in enumerate(unique_classes):
            idx = np.where(y == cls)[0][0]
            series = X[idx, 0, :] if X.ndim == 3 else X[idx]
            sample_fig.add_trace(
                go.Scatter(y=series, mode='lines', name=f"Class {cls}", line=dict(color=colors[i]))
            )
    sample_fig.update_layout(
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return (
        X.tolist(),
        y.tolist() if y is not None else None,
        overview,
        class_fig,
        sample_fig,
    )


@callback(
    Output("clf-estimator-info", "children"),
    Output("clf-params-container", "children"),
    Input("clf-estimator-select", "value"),
)
def update_estimator_info(estimator_name):
    if not estimator_name:
        return "", ""

    discovery = get_aeon_discovery()
    docstring = discovery.get_estimator_docstring(estimator_name, "classifier")
    params = discovery.get_estimator_params(estimator_name, "classifier")

    # Short description
    desc = docstring.split('\n')[0] if docstring else ""

    # Basic params display
    param_inputs = []
    for name, info in list(params.items())[:3]:
        if name in ("n_jobs", "verbose", "random_state"):
            continue
        param_inputs.append(
            html.Div(
                [
                    dbc.Label(name.replace("_", " ").title(), className="small"),
                    dbc.Input(
                        id={"type": "clf-param", "name": name},
                        type="text",
                        value=str(info.get("default", "")),
                        size="sm",
                    ),
                ],
                className="mb-2",
            )
        )

    return desc[:100] + "..." if len(desc) > 100 else desc, html.Div(param_inputs)


@callback(
    Output("clf-status-alert", "is_open"),
    Output("clf-status-alert", "children"),
    Output("clf-status-alert", "color"),
    Output("clf-metrics-container", "children"),
    Output("clf-confusion-container", "children"),
    Output("clf-logs-container", "children"),
    Output("clf-result", "data"),
    Input("btn-clf-run", "n_clicks"),
    State("clf-data-X", "data"),
    State("clf-data-y", "data"),
    State("clf-estimator-select", "value"),
    State("clf-test-size", "value"),
    State("clf-random-state", "value"),
    prevent_initial_call=True,
)
def run_classification(n_clicks, X_data, y_data, estimator, test_size, random_state):
    if X_data is None or y_data is None:
        return True, "Please load a dataset first", "warning", None, None, None, None

    X = np.array(X_data)
    y = np.array(y_data)

    runner = TaskRunner()

    result = runner.run_classification(
        X=X,
        y=y,
        estimator_name=estimator or "KNeighborsTimeSeriesClassifier",
        params={},
        test_size=test_size or 0.2,
        random_state=random_state or 42,
    )

    if result.success:
        metrics_display = create_metrics_display(result.metrics, "classification")
        confusion_display = create_confusion_matrix(np.array(result.confusion_matrix))
        logs_display = create_run_logs(result.logs)

        return (
            True,
            f"Training completed in {result.train_time:.2f}s",
            "success",
            metrics_display,
            confusion_display,
            logs_display,
            result.to_dict(),
        )
    else:
        return (
            True,
            f"Error: {result.error_message}",
            "danger",
            None,
            None,
            create_run_logs(result.logs) if result.logs else None,
            None,
        )


@callback(
    Output("clf-code-modal", "is_open"),
    Output("clf-code-modal-code-content", "children"),
    Input("btn-clf-export", "n_clicks"),
    Input("clf-code-modal-close-btn", "n_clicks"),
    State("clf-estimator-select", "value"),
    State("clf-dataset-select", "value"),
    State("clf-code-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_code_modal(n_export, n_close, estimator, dataset, is_open):
    triggered = ctx.triggered_id

    if triggered == "btn-clf-export":
        exporter = CodeExporter()
        discovery = get_aeon_discovery()

        est_class = discovery.get_estimator_class(estimator or "KNeighborsTimeSeriesClassifier", "classifier")
        module = est_class.__module__ if est_class else "aeon.classification"

        code = exporter.generate_classification_code(
            estimator_name=estimator or "KNeighborsTimeSeriesClassifier",
            estimator_module=module,
            params={"random_state": 42},
            dataset_name=dataset.replace("builtin:", "") if dataset and dataset.startswith("builtin:") else None,
        )

        return True, html.Pre(code)

    return False, ""

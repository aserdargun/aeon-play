"""
Data Studio page for loading, previewing, and preparing datasets.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import base64
import io
import json
import uuid

from aeon_play.services.datasets import get_dataset_service, DatasetInfo
from aeon_play.components.data_preview import (
    create_data_preview,
    create_data_warnings,
    create_series_selector,
)
from aeon_play.components.learning_section import create_learning_section
from aeon_play.components.level_gate import level_gate, get_level_features

dash.register_page(__name__, path="/data", name="Data Studio", title="aeon-play | Data Studio")


def create_dataset_selector():
    """Create the built-in dataset selector."""
    service = get_dataset_service()
    datasets = service.get_available_datasets()

    # Classification datasets
    classification_options = []
    if "classification" in datasets:
        for ds in datasets["classification"].get("univariate", []):
            classification_options.append({"label": f"{ds} (Univariate)", "value": f"classification:{ds}"})
        for ds in datasets["classification"].get("multivariate", []):
            classification_options.append({"label": f"{ds} (Multivariate)", "value": f"classification:{ds}"})

    # Regression datasets
    regression_options = [
        {"label": ds, "value": f"regression:{ds}"}
        for ds in datasets.get("regression", [])
    ]

    # Forecasting datasets
    forecasting_options = [
        {"label": ds, "value": f"forecasting:{ds}"}
        for ds in datasets.get("forecasting", [])
    ]

    return dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className="fas fa-database me-2"), "Built-in Datasets"]
            ),
            dbc.CardBody(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                [
                                    dbc.Select(
                                        id="dataset-classification-select",
                                        options=classification_options,
                                        placeholder="Select a classification dataset...",
                                        className="mt-2",
                                    ),
                                ],
                                label="Classification",
                            ),
                            dbc.Tab(
                                [
                                    dbc.Select(
                                        id="dataset-regression-select",
                                        options=regression_options,
                                        placeholder="Select a regression dataset...",
                                        className="mt-2",
                                    ),
                                ],
                                label="Regression",
                            ),
                            dbc.Tab(
                                [
                                    dbc.Select(
                                        id="dataset-forecasting-select",
                                        options=forecasting_options,
                                        placeholder="Select a forecasting dataset...",
                                        className="mt-2",
                                    ),
                                ],
                                label="Forecasting",
                            ),
                        ],
                        id="dataset-type-tabs",
                        active_tab="tab-0",
                    ),
                    html.Div(className="mt-3"),
                    dbc.Button(
                        [html.I(className="fas fa-download me-2"), "Load Dataset"],
                        id="btn-load-builtin",
                        color="primary",
                        className="w-100",
                    ),
                ]
            ),
        ],
        className="mb-4",
    )


def create_file_upload():
    """Create file upload section."""
    return dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className="fas fa-upload me-2"), "Upload Data"]
            ),
            dbc.CardBody(
                [
                    dcc.Upload(
                        id="data-upload",
                        children=html.Div(
                            [
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                                html.P("Drag and drop or click to upload"),
                                html.Small(
                                    "Supports: CSV, .ts, .tsf files",
                                    className="text-muted",
                                ),
                            ],
                            className="text-center py-4",
                        ),
                        style={
                            "border": "2px dashed #ddd",
                            "borderRadius": "0.5rem",
                            "cursor": "pointer",
                        },
                        multiple=False,
                    ),
                    html.Div(id="upload-status", className="mt-2"),
                ]
            ),
        ],
        className="mb-4",
    )


def create_data_options(level: str):
    """Create data options based on level."""
    features = get_level_features(level)

    options = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Train/Test Split", className="small"),
                        dbc.Input(
                            id="data-test-size",
                            type="number",
                            min=0.1,
                            max=0.5,
                            step=0.05,
                            value=0.2,
                            size="sm",
                            disabled=not features.get("train_test_split", False),
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Label("Random State", className="small"),
                        dbc.Input(
                            id="data-random-state",
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
    ]

    if features.get("multivariate", False):
        options.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Unequal Length Strategy", className="small"),
                            dbc.Select(
                                id="data-unequal-strategy",
                                options=[
                                    {"label": "Pad with zeros", "value": "pad"},
                                    {"label": "Truncate to min", "value": "truncate"},
                                    {"label": "Resample", "value": "resample"},
                                ],
                                value="pad",
                                size="sm",
                            ),
                        ],
                    ),
                ],
                className="mb-2",
            )
        )

    return html.Div(options)


def create_session_datasets():
    """Create session datasets panel."""
    return dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className="fas fa-folder-open me-2"), "Session Datasets"]
            ),
            dbc.CardBody(
                [
                    html.Div(id="session-datasets-list"),
                    html.Small(
                        "Datasets loaded in this session",
                        className="text-muted",
                    ),
                ]
            ),
        ],
        className="mb-4",
    )


# Page layout
layout = html.Div(
    [
        # Learning section
        create_learning_section(
            title="Data Studio",
            content=(
                "Start your time series journey by loading data. You can use "
                "built-in datasets from the aeon library or upload your own files. "
                "This is where you'll prepare and validate your data before analysis."
            ),
            tips=[
                "Built-in datasets are great for learning - they're curated and ready to use",
                "CSV files should have time series as rows",
                "aeon's .ts format is ideal for time series collections",
            ],
        ),

        dbc.Row(
            [
                # Left panel - data sources
                dbc.Col(
                    [
                        create_dataset_selector(),
                        create_file_upload(),
                        html.Div(id="data-options-container"),
                        create_session_datasets(),
                    ],
                    md=4,
                ),

                # Right panel - preview
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.I(className="fas fa-eye me-2"),
                                        "Data Preview",
                                        dbc.Badge(
                                            id="current-dataset-badge",
                                            color="primary",
                                            className="ms-2",
                                        ),
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-preview",
                                            type="default",
                                            children=html.Div(id="data-preview-container"),
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(id="data-warnings-container", className="mt-3"),

                        # Export options
                        dbc.Card(
                            [
                                dbc.CardHeader("Export"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Button(
                                                        [
                                                            html.I(className="fas fa-save me-2"),
                                                            "Save to Session",
                                                        ],
                                                        id="btn-save-session",
                                                        color="primary",
                                                        size="sm",
                                                        className="w-100",
                                                    ),
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        [
                                                            html.I(className="fas fa-download me-2"),
                                                            "Download .npz",
                                                        ],
                                                        id="btn-download-npz",
                                                        color="secondary",
                                                        size="sm",
                                                        className="w-100",
                                                    ),
                                                    width=6,
                                                ),
                                            ],
                                        ),
                                        dcc.Download(id="download-dataset"),
                                    ]
                                ),
                            ],
                            className="mt-3",
                        ),
                    ],
                    md=8,
                ),
            ],
        ),

        # Hidden stores
        dcc.Store(id="current-data-X", storage_type="memory"),
        dcc.Store(id="current-data-y", storage_type="memory"),
        dcc.Store(id="current-data-info", storage_type="memory"),
    ],
)


# Callbacks
@callback(
    Output("data-options-container", "children"),
    Input("user-level", "data"),
)
def update_data_options(level):
    """Update data options based on user level."""
    return create_data_options(level or "beginner")


@callback(
    Output("data-preview-container", "children"),
    Output("data-warnings-container", "children"),
    Output("current-dataset-badge", "children"),
    Output("current-data-X", "data"),
    Output("current-data-y", "data"),
    Output("current-data-info", "data"),
    Input("btn-load-builtin", "n_clicks"),
    Input("data-upload", "contents"),
    State("dataset-classification-select", "value"),
    State("dataset-regression-select", "value"),
    State("dataset-forecasting-select", "value"),
    State("data-upload", "filename"),
    State("user-level", "data"),
    prevent_initial_call=True,
)
def load_and_preview_data(
    n_clicks, upload_contents, cls_ds, reg_ds, fc_ds, filename, level
):
    """Load dataset and show preview."""
    service = get_dataset_service()
    triggered_id = ctx.triggered_id

    try:
        if triggered_id == "btn-load-builtin":
            # Determine which dataset was selected
            selected = cls_ds or reg_ds or fc_ds
            if not selected:
                return (
                    html.Div("Please select a dataset first", className="text-muted"),
                    None,
                    "No dataset",
                    None, None, None,
                )

            task_type, name = selected.split(":", 1)
            X, y, info = service.load_builtin_dataset(name, task_type)

        elif triggered_id == "data-upload" and upload_contents:
            # Parse uploaded file
            content_type, content_string = upload_contents.split(",")
            decoded = base64.b64decode(content_string)
            X, y, info = service.load_from_file(decoded, filename)

        else:
            return (
                html.Div("Select a dataset or upload a file", className="text-muted"),
                None,
                "No dataset",
                None, None, None,
            )

        # Create preview
        preview = create_data_preview(X, y, info.to_dict())
        warnings = create_data_warnings(info.to_dict())

        # Serialize data for storage
        X_json = X.tolist() if isinstance(X, np.ndarray) else None
        y_json = y.tolist() if isinstance(y, np.ndarray) else None

        return (
            preview,
            warnings,
            info.name,
            X_json,
            y_json,
            info.to_dict(),
        )

    except Exception as e:
        error_msg = html.Div(
            [
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                f"Error loading data: {str(e)}",
            ],
            className="text-danger",
        )
        return error_msg, None, "Error", None, None, None


@callback(
    Output("upload-status", "children"),
    Input("data-upload", "filename"),
    prevent_initial_call=True,
)
def update_upload_status(filename):
    """Show uploaded filename."""
    if filename:
        return dbc.Alert(
            [html.I(className="fas fa-file me-2"), f"Uploaded: {filename}"],
            color="success",
            className="mb-0 py-2",
        )
    return None


@callback(
    Output("session-datasets", "data", allow_duplicate=True),
    Output("session-datasets-list", "children"),
    Input("btn-save-session", "n_clicks"),
    State("current-data-X", "data"),
    State("current-data-y", "data"),
    State("current-data-info", "data"),
    State("session-datasets", "data"),
    prevent_initial_call=True,
)
def save_to_session(n_clicks, X_data, y_data, info, current_datasets):
    """Save current dataset to session."""
    if not n_clicks or X_data is None:
        return dash.no_update, dash.no_update

    if current_datasets is None:
        current_datasets = {}

    if info:
        key = info.get("name", f"dataset_{len(current_datasets)}")
        current_datasets[key] = {
            "X": X_data,
            "y": y_data,
            "info": info,
        }

    # Create list of saved datasets
    dataset_list = [
        dbc.ListGroupItem(
            [
                html.I(className="fas fa-database me-2"),
                name,
                dbc.Badge(
                    current_datasets[name]["info"].get("task_type", "unknown"),
                    color="secondary",
                    className="ms-auto",
                ),
            ],
            className="d-flex align-items-center",
        )
        for name in current_datasets.keys()
    ]

    return current_datasets, dbc.ListGroup(dataset_list, flush=True)


@callback(
    Output("download-dataset", "data"),
    Input("btn-download-npz", "n_clicks"),
    State("current-data-X", "data"),
    State("current-data-y", "data"),
    State("current-data-info", "data"),
    prevent_initial_call=True,
)
def download_dataset(n_clicks, X_data, y_data, info):
    """Download dataset as .npz file."""
    if not n_clicks or X_data is None:
        return dash.no_update

    X = np.array(X_data)
    y = np.array(y_data) if y_data else None
    name = info.get("name", "dataset") if info else "dataset"

    buffer = io.BytesIO()
    if y is not None:
        np.savez(buffer, X=X, y=y)
    else:
        np.savez(buffer, X=X)

    return dcc.send_bytes(buffer.getvalue(), f"{name}.npz")

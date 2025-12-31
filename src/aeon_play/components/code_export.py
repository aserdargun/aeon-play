"""
Code export modal component for generating reproducible scripts.
"""

from typing import Optional
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_code_export_modal(
    modal_id: str,
    title: str = "Export Code",
) -> dbc.Modal:
    """
    Create a code export modal component.

    Args:
        modal_id: Unique ID for the modal
        title: Modal title

    Returns:
        Modal component
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(
                [
                    html.P(
                        "Copy this code to reproduce your analysis:",
                        className="text-muted mb-3",
                    ),
                    html.Div(
                        id=f"{modal_id}-code-content",
                        className="code-export",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-copy me-2"), "Copy to Clipboard"],
                        id=f"{modal_id}-copy-btn",
                        color="primary",
                        className="mt-3",
                    ),
                    html.Div(
                        id=f"{modal_id}-copy-feedback",
                        className="text-success mt-2 small",
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        [html.I(className="fas fa-download me-2"), "Download .py"],
                        id=f"{modal_id}-download-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Close",
                        id=f"{modal_id}-close-btn",
                        color="outline-secondary",
                    ),
                ]
            ),
        ],
        id=modal_id,
        size="lg",
        is_open=False,
    )


def create_code_block(code: str) -> html.Div:
    """
    Create a code block with syntax highlighting placeholder.

    Args:
        code: Python code string

    Returns:
        Code block component
    """
    return html.Div(
        html.Pre(
            html.Code(code, className="language-python"),
            className="mb-0",
        ),
        className="code-export",
    )


def create_export_button(
    button_id: str,
    label: str = "Generate Code",
    icon: str = "fa-code",
    color: str = "outline-primary",
    size: str = "sm",
) -> dbc.Button:
    """
    Create an export code button.

    Args:
        button_id: Button ID
        label: Button label
        icon: FontAwesome icon
        color: Button color
        size: Button size

    Returns:
        Button component
    """
    return dbc.Button(
        [html.I(className=f"fas {icon} me-1"), label],
        id=button_id,
        color=color,
        size=size,
    )


def create_download_component(download_id: str) -> dcc.Download:
    """
    Create a download component for file export.

    Args:
        download_id: Component ID

    Returns:
        Download component
    """
    return dcc.Download(id=download_id)


def format_code_for_display(code: str) -> html.Pre:
    """
    Format code for display with proper styling.

    Args:
        code: Raw code string

    Returns:
        Formatted pre element
    """
    # Add line numbers
    lines = code.split('\n')
    numbered_lines = []

    for i, line in enumerate(lines, 1):
        numbered_lines.append(
            html.Div(
                [
                    html.Span(
                        f"{i:3d} ",
                        style={
                            "color": "#666",
                            "userSelect": "none",
                            "marginRight": "10px",
                        },
                    ),
                    line,
                ],
                style={"whiteSpace": "pre"},
            )
        )

    return html.Pre(
        numbered_lines,
        className="mb-0",
        style={
            "fontFamily": "'Fira Code', 'Consolas', monospace",
            "fontSize": "0.85rem",
            "lineHeight": "1.5",
        },
    )


def create_pipeline_builder_modal(modal_id: str) -> dbc.Modal:
    """
    Create a pipeline builder modal for advanced users.

    Args:
        modal_id: Unique ID for the modal

    Returns:
        Modal component
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Build Pipeline")),
            dbc.ModalBody(
                [
                    html.P(
                        "Chain transformers and estimators into a pipeline:",
                        className="text-muted mb-3",
                    ),
                    # Pipeline steps
                    html.Div(
                        id=f"{modal_id}-steps",
                        className="mb-3",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-plus me-2"), "Add Step"],
                        id=f"{modal_id}-add-step",
                        color="outline-primary",
                        size="sm",
                    ),
                    html.Hr(),
                    # Generated code preview
                    html.H6("Generated Code"),
                    html.Div(
                        id=f"{modal_id}-code-preview",
                        className="code-export",
                        style={"maxHeight": "200px", "overflow": "auto"},
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Apply Pipeline",
                        id=f"{modal_id}-apply-btn",
                        color="primary",
                    ),
                    dbc.Button(
                        "Cancel",
                        id=f"{modal_id}-cancel-btn",
                        color="outline-secondary",
                    ),
                ]
            ),
        ],
        id=modal_id,
        size="lg",
        is_open=False,
    )

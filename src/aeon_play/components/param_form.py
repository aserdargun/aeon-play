"""
Parameter form component for estimator configuration.
"""

from typing import Any, Dict, List, Optional, Tuple
import json

from dash import html, dcc, callback, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc


def create_param_form(
    params: Dict[str, Dict],
    prefix: str,
    include_random_state: bool = True,
    param_docs: Optional[Dict[str, str]] = None,
) -> html.Div:
    """
    Create a parameter form from a params dictionary.

    Args:
        params: Dict of param_name -> {default, type, required}
        prefix: Prefix for component IDs
        include_random_state: Whether to include random_state
        param_docs: Optional dict of param_name -> description

    Returns:
        Div containing form elements
    """
    if param_docs is None:
        param_docs = {}

    form_elements = []

    for param_name, param_info in params.items():
        if param_name in ("n_jobs", "verbose", "callbacks"):
            continue

        default_val = param_info.get("default")
        param_type = param_info.get("type", "")
        description = param_docs.get(param_name, "")

        elem = _create_single_param(
            param_name=param_name,
            default_value=default_val,
            param_type=param_type,
            description=description,
            prefix=prefix,
        )

        if elem:
            form_elements.append(elem)

    # Add random state control
    if include_random_state:
        form_elements.append(
            html.Div(
                [
                    dbc.Label(
                        [
                            "Random State",
                            html.I(
                                className="fas fa-question-circle tooltip-icon ms-1",
                                id=f"{prefix}-random_state-tip",
                            ),
                        ],
                        className="fw-medium small",
                    ),
                    dbc.Tooltip(
                        "Seed for reproducibility",
                        target=f"{prefix}-random_state-tip",
                    ),
                    dbc.Input(
                        id={"type": f"{prefix}-param", "name": "random_state"},
                        type="number",
                        value=42,
                        size="sm",
                    ),
                ],
                className="mb-2",
            )
        )

    return html.Div(
        form_elements,
        className="param-form",
        id=f"{prefix}-form-container",
    )


def _create_single_param(
    param_name: str,
    default_value: Any,
    param_type: str,
    description: str,
    prefix: str,
) -> Optional[html.Div]:
    """Create a single parameter input element."""
    component_id = {"type": f"{prefix}-param", "name": param_name}
    tooltip_id = f"{prefix}-{param_name}-tip"

    # Determine input type
    if isinstance(default_value, bool):
        input_elem = dbc.Checkbox(
            id=component_id,
            value=default_value,
            label="",
        )
    elif isinstance(default_value, int) and not isinstance(default_value, bool):
        input_elem = dbc.Input(
            id=component_id,
            type="number",
            value=default_value if default_value is not None else "",
            step=1,
            size="sm",
        )
    elif isinstance(default_value, float):
        input_elem = dbc.Input(
            id=component_id,
            type="number",
            value=default_value if default_value is not None else "",
            step=0.01,
            size="sm",
        )
    elif isinstance(default_value, (list, dict)):
        input_elem = dbc.Textarea(
            id=component_id,
            value=json.dumps(default_value) if default_value else "",
            size="sm",
            style={"fontFamily": "monospace", "fontSize": "0.8rem"},
            rows=2,
        )
    else:
        input_elem = dbc.Input(
            id=component_id,
            type="text",
            value=str(default_value) if default_value is not None else "",
            size="sm",
        )

    label_text = param_name.replace("_", " ").title()

    return html.Div(
        [
            dbc.Label(
                [
                    label_text,
                    html.I(
                        className="fas fa-question-circle tooltip-icon ms-1",
                        id=tooltip_id,
                    ) if description else None,
                ],
                className="fw-medium small",
            ),
            dbc.Tooltip(
                description[:150] + "..." if len(description) > 150 else description,
                target=tooltip_id,
            ) if description else None,
            input_elem,
        ],
        className="mb-2",
    )


def get_param_values(
    form_values: List[Any],
    param_names: List[str],
    param_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Extract and parse parameter values from form inputs.

    Args:
        form_values: List of values from pattern-matching callback
        param_names: List of parameter names corresponding to values
        param_types: Optional type information for parsing

    Returns:
        Dict of param_name -> parsed_value
    """
    if param_types is None:
        param_types = {}

    result = {}

    for name, value in zip(param_names, form_values):
        if value is None or value == "":
            result[name] = None
            continue

        if value == "None":
            result[name] = None
            continue

        type_hint = param_types.get(name, "")

        try:
            if isinstance(value, bool):
                result[name] = value
            elif "int" in str(type_hint).lower():
                result[name] = int(float(value))
            elif "float" in str(type_hint).lower():
                result[name] = float(value)
            elif isinstance(value, str):
                if value.startswith(("[", "{")):
                    result[name] = json.loads(value)
                elif value.lower() == "true":
                    result[name] = True
                elif value.lower() == "false":
                    result[name] = False
                else:
                    # Try numeric conversion
                    try:
                        if "." in value:
                            result[name] = float(value)
                        else:
                            result[name] = int(value)
                    except ValueError:
                        result[name] = value
            else:
                result[name] = value

        except (ValueError, json.JSONDecodeError):
            result[name] = value

    return result


def create_estimator_selector(
    estimators: List[Dict],
    selected: Optional[str] = None,
    prefix: str = "est",
) -> html.Div:
    """
    Create an estimator selection component.

    Args:
        estimators: List of estimator info dicts
        selected: Currently selected estimator name
        prefix: ID prefix

    Returns:
        Estimator selector component
    """
    options = [
        {"label": est["name"], "value": est["name"]}
        for est in estimators
    ]

    return html.Div(
        [
            dbc.Label("Select Algorithm", className="fw-medium"),
            dbc.Select(
                id=f"{prefix}-selector",
                options=options,
                value=selected or (options[0]["value"] if options else None),
                className="mb-2",
            ),
            html.Div(id=f"{prefix}-info", className="small text-muted"),
        ],
    )


def create_reset_button(prefix: str) -> dbc.Button:
    """Create a reset to defaults button."""
    return dbc.Button(
        [html.I(className="fas fa-undo me-1"), "Reset"],
        id=f"{prefix}-reset-btn",
        color="outline-secondary",
        size="sm",
        className="me-2",
    )


def create_run_button(prefix: str, loading: bool = False) -> dbc.Button:
    """Create a run button."""
    return dbc.Button(
        [
            dbc.Spinner(size="sm", className="me-2") if loading else html.I(className="fas fa-play me-1"),
            "Run",
        ],
        id=f"{prefix}-run-btn",
        color="primary",
        disabled=loading,
    )

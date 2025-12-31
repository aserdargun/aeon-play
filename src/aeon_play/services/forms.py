"""
Dynamic form generation for estimator parameters.

Creates Dash components from estimator parameter schemas.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import inspect

from dash import html, dcc
import dash_bootstrap_components as dbc

from .aeon_discovery import get_aeon_discovery


class FormGenerator:
    """Generates Dash form components from estimator parameters."""

    # Parameters to exclude from forms
    EXCLUDED_PARAMS = {
        "n_jobs", "verbose", "callbacks", "random_state",  # Handled separately
    }

    # Parameter type hints
    TYPE_HINTS = {
        int: "number",
        float: "number",
        bool: "checkbox",
        str: "text",
        list: "json",
        dict: "json",
        type(None): "text",
    }

    def __init__(self):
        """Initialize the form generator."""
        self.discovery = get_aeon_discovery()

    def generate_form(
        self,
        estimator_name: str,
        estimator_type: str,
        prefix: str = "param",
        include_random_state: bool = True,
    ) -> Tuple[List, Dict[str, Any]]:
        """
        Generate form components for an estimator.

        Args:
            estimator_name: Name of the estimator
            estimator_type: Type (classifier, regressor, etc.)
            prefix: Prefix for component IDs
            include_random_state: Whether to include random_state control

        Returns:
            Tuple of (list of form components, dict of default values)
        """
        params = self.discovery.get_estimator_params(estimator_name, estimator_type)
        param_docs = self.discovery.get_param_docstrings(estimator_name, estimator_type)

        components = []
        defaults = {}

        for param_name, param_info in params.items():
            if param_name in self.EXCLUDED_PARAMS and param_name != "random_state":
                continue

            if param_name == "random_state" and not include_random_state:
                continue

            default_val = param_info.get("default")
            param_type = param_info.get("type")
            description = param_docs.get(param_name, "")

            # Store default
            defaults[param_name] = default_val

            # Create form component
            component = self._create_param_component(
                param_name=param_name,
                param_type=param_type,
                default_value=default_val,
                description=description,
                prefix=prefix,
            )

            if component:
                components.append(component)

        # Add random_state at the end if requested
        if include_random_state:
            defaults["random_state"] = 42
            components.append(
                self._create_random_state_component(prefix)
            )

        return components, defaults

    def _create_param_component(
        self,
        param_name: str,
        param_type: Optional[str],
        default_value: Any,
        description: str,
        prefix: str,
    ):
        """Create a single parameter form component."""
        component_id = f"{prefix}-{param_name}"
        tooltip_id = f"{component_id}-tooltip"

        # Determine input type
        input_type = self._infer_input_type(param_type, default_value)

        # Create label with tooltip
        label = dbc.Label(
            [
                param_name.replace("_", " ").title(),
                html.I(
                    className="fas fa-question-circle tooltip-icon ms-1",
                    id=tooltip_id,
                ) if description else None,
            ],
            html_for=component_id,
            className="fw-medium",
        )

        tooltip = dbc.Tooltip(
            description[:200] + "..." if len(description) > 200 else description,
            target=tooltip_id,
            placement="right",
        ) if description else None

        # Create input based on type
        if input_type == "checkbox":
            input_elem = dbc.Checkbox(
                id=component_id,
                value=bool(default_value) if default_value is not None else False,
                className="form-check",
            )

        elif input_type == "number":
            # Determine if int or float
            if isinstance(default_value, float) and not isinstance(default_value, bool):
                step = 0.01
            else:
                step = 1

            input_elem = dbc.Input(
                id=component_id,
                type="number",
                value=default_value if default_value is not None else "",
                step=step,
                className="form-control form-control-sm",
            )

        elif input_type == "select":
            # For enum-like types
            options = self._extract_options(param_type, default_value)
            input_elem = dbc.Select(
                id=component_id,
                options=options,
                value=str(default_value) if default_value is not None else "",
                className="form-select form-select-sm",
            )

        elif input_type == "json":
            # For complex types like list or dict
            input_elem = dbc.Textarea(
                id=component_id,
                value=json.dumps(default_value) if default_value is not None else "",
                className="form-control form-control-sm",
                style={"fontFamily": "monospace", "fontSize": "0.85rem"},
                rows=2,
            )

        else:
            # Default to text input
            input_elem = dbc.Input(
                id=component_id,
                type="text",
                value=str(default_value) if default_value is not None else "",
                className="form-control form-control-sm",
            )

        # Combine into form group
        return html.Div(
            [
                label,
                tooltip,
                input_elem,
            ],
            className="param-group mb-2",
        )

    def _create_random_state_component(self, prefix: str):
        """Create the random_state component."""
        return html.Div(
            [
                dbc.Label(
                    [
                        "Random State",
                        html.I(
                            className="fas fa-question-circle tooltip-icon ms-1",
                            id=f"{prefix}-random_state-tooltip",
                        ),
                    ],
                    html_for=f"{prefix}-random_state",
                    className="fw-medium",
                ),
                dbc.Tooltip(
                    "Set a seed for reproducibility. Use the same value to get consistent results.",
                    target=f"{prefix}-random_state-tooltip",
                    placement="right",
                ),
                dbc.Input(
                    id=f"{prefix}-random_state",
                    type="number",
                    value=42,
                    step=1,
                    className="form-control form-control-sm",
                ),
            ],
            className="param-group mb-2",
        )

    def _infer_input_type(
        self,
        param_type: Optional[str],
        default_value: Any,
    ) -> str:
        """Infer the appropriate input type."""
        # Check by default value type
        if isinstance(default_value, bool):
            return "checkbox"
        elif isinstance(default_value, int) and not isinstance(default_value, bool):
            return "number"
        elif isinstance(default_value, float):
            return "number"
        elif isinstance(default_value, (list, dict)):
            return "json"

        # Check by type annotation
        if param_type:
            type_str = str(param_type).lower()
            if "bool" in type_str:
                return "checkbox"
            elif "int" in type_str or "float" in type_str:
                return "number"
            elif "literal" in type_str or "union" in type_str:
                return "select"

        return "text"

    def _extract_options(
        self,
        param_type: Optional[str],
        default_value: Any,
    ) -> List[Dict]:
        """Extract options for select-type inputs."""
        options = []

        if param_type and "Literal" in str(param_type):
            # Try to parse Literal type
            import re
            matches = re.findall(r"'([^']+)'", str(param_type))
            for match in matches:
                options.append({"label": match, "value": match})

        if not options and default_value is not None:
            options.append({"label": str(default_value), "value": str(default_value)})

        return options

    @staticmethod
    def parse_form_values(
        form_values: Dict[str, Any],
        param_types: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parse form values back to appropriate Python types.

        Args:
            form_values: Dict of param_name -> form_value
            param_types: Dict of param_name -> type info

        Returns:
            Dict of param_name -> parsed_value
        """
        parsed = {}

        for name, value in form_values.items():
            if value is None or value == "":
                parsed[name] = None
                continue

            if value == "None":
                parsed[name] = None
                continue

            type_info = param_types.get(name, {})
            expected_type = type_info.get("type", "")

            try:
                if isinstance(value, bool):
                    parsed[name] = value
                elif expected_type and "int" in str(expected_type).lower():
                    parsed[name] = int(float(value))
                elif expected_type and "float" in str(expected_type).lower():
                    parsed[name] = float(value)
                elif expected_type and "bool" in str(expected_type).lower():
                    parsed[name] = value in (True, "true", "True", "1", 1)
                elif isinstance(value, str) and value.startswith(("[", "{")):
                    parsed[name] = json.loads(value)
                else:
                    parsed[name] = value
            except (ValueError, json.JSONDecodeError):
                parsed[name] = value

        return parsed


def create_quick_form(
    params: Dict[str, Any],
    prefix: str = "quick",
) -> List:
    """
    Create a quick form from a dict of param_name -> default_value.

    Useful for simpler cases where full estimator discovery isn't needed.
    """
    components = []
    generator = FormGenerator()

    for name, default in params.items():
        component = generator._create_param_component(
            param_name=name,
            param_type=None,
            default_value=default,
            description="",
            prefix=prefix,
        )
        if component:
            components.append(component)

    return components


def create_preset_buttons(
    presets: List[Dict[str, Any]],
    button_id_prefix: str,
) -> List:
    """
    Create preset buttons for quick configuration.

    Args:
        presets: List of dicts with 'name', 'description', 'params'
        button_id_prefix: Prefix for button IDs

    Returns:
        List of button components
    """
    buttons = []

    for i, preset in enumerate(presets):
        buttons.append(
            dbc.Button(
                [
                    html.I(className="fas fa-magic me-1"),
                    preset.get("name", f"Preset {i+1}"),
                ],
                id=f"{button_id_prefix}-{i}",
                color="outline-secondary",
                size="sm",
                className="preset-btn",
                title=preset.get("description", ""),
            )
        )

    return html.Div(
        [
            html.Small("Quick presets:", className="text-muted me-2"),
            *buttons,
        ],
        className="mb-3",
    )

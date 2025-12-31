"""
Level gating component for progressive disclosure.
"""

from typing import Any, Dict, List, Optional, Union
from dash import html
import dash_bootstrap_components as dbc


# Level hierarchy
LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}


def level_gate(
    content: Any,
    min_level: str,
    current_level: str,
    hidden_message: Optional[str] = None,
) -> Any:
    """
    Gate content based on user level.

    Args:
        content: The content to show/hide
        min_level: Minimum level required to see content
        current_level: User's current level
        hidden_message: Optional message to show when hidden

    Returns:
        Content or hidden message or None
    """
    required = LEVEL_ORDER.get(min_level, 0)
    current = LEVEL_ORDER.get(current_level, 0)

    if current >= required:
        return content

    if hidden_message:
        return html.Div(
            [
                html.I(className="fas fa-lock me-2"),
                html.Small(hidden_message, className="text-muted"),
            ],
            className="text-center py-2 bg-light rounded",
        )

    return None


def get_visible_items(
    items: List[Dict],
    level: str,
    level_key: str = "level",
) -> List[Dict]:
    """
    Filter items based on user level.

    Args:
        items: List of items with level info
        level: User's current level
        level_key: Key in item dict that contains level info

    Returns:
        Filtered list of items
    """
    current = LEVEL_ORDER.get(level, 0)

    return [
        item for item in items
        if LEVEL_ORDER.get(item.get(level_key, "beginner"), 0) <= current
    ]


def create_level_indicator(current_level: str) -> html.Div:
    """
    Create a visual level indicator.

    Args:
        current_level: User's current level

    Returns:
        Level indicator component
    """
    levels = [
        ("beginner", "Beginner", "fa-seedling", "success"),
        ("intermediate", "Intermediate", "fa-tree", "warning"),
        ("advanced", "Advanced", "fa-mountain", "danger"),
    ]

    indicators = []
    current_idx = LEVEL_ORDER.get(current_level, 0)

    for idx, (level, label, icon, color) in enumerate(levels):
        is_active = idx <= current_idx

        indicators.append(
            html.Div(
                [
                    html.I(
                        className=f"fas {icon}",
                        style={"opacity": 1 if is_active else 0.3},
                    ),
                    html.Small(
                        label,
                        className=f"ms-1 {'fw-bold' if level == current_level else ''}",
                        style={"opacity": 1 if is_active else 0.5},
                    ),
                ],
                className=f"d-flex align-items-center {'text-' + color if is_active else 'text-muted'}",
            )
        )

    return html.Div(
        [
            html.Small("Your Level:", className="text-muted me-2"),
            *[html.Span(ind, className="me-3") for ind in indicators],
        ],
        className="d-flex align-items-center",
    )


def create_level_locked_overlay(
    content: Any,
    required_level: str,
    current_level: str,
) -> html.Div:
    """
    Create content with a locked overlay for higher levels.

    Args:
        content: The content (will be blurred/overlaid)
        required_level: Level required to access
        current_level: User's current level

    Returns:
        Content with overlay if locked
    """
    required = LEVEL_ORDER.get(required_level, 0)
    current = LEVEL_ORDER.get(current_level, 0)

    if current >= required:
        return content

    return html.Div(
        [
            html.Div(
                content,
                style={
                    "filter": "blur(3px)",
                    "pointerEvents": "none",
                    "opacity": 0.5,
                },
            ),
            html.Div(
                [
                    html.I(className="fas fa-lock fa-2x mb-2"),
                    html.P(
                        f"Switch to {required_level.title()} level to unlock",
                        className="mb-0",
                    ),
                ],
                className="position-absolute top-50 start-50 translate-middle text-center bg-white p-3 rounded shadow",
                style={"zIndex": 10},
            ),
        ],
        className="position-relative",
    )


def get_level_features(level: str) -> Dict[str, bool]:
    """
    Get feature flags for a given level.

    Args:
        level: User level

    Returns:
        Dict of feature_name -> enabled
    """
    features = {
        # Beginner features
        "builtin_datasets": True,
        "csv_upload": True,
        "basic_preview": True,
        "simple_estimators": True,
        "basic_metrics": True,

        # Intermediate features
        "ts_file_upload": level in ("intermediate", "advanced"),
        "train_test_split": level in ("intermediate", "advanced"),
        "cross_validation": level in ("intermediate", "advanced"),
        "more_estimators": level in ("intermediate", "advanced"),
        "pipelines": level in ("intermediate", "advanced"),
        "intermediate_viz": level in ("intermediate", "advanced"),

        # Advanced features
        "multivariate": level == "advanced",
        "unequal_length": level == "advanced",
        "custom_metrics": level == "advanced",
        "benchmarking": level == "advanced",
        "code_export": level == "advanced",
        "all_estimators": level == "advanced",
        "hyperparameter_tuning": level == "advanced",
    }

    return features


def get_level_description(level: str) -> str:
    """Get a description of what the level offers."""
    descriptions = {
        "beginner": (
            "Perfect for getting started! You'll have access to curated datasets, "
            "simple algorithms, and guided workflows. Focus on understanding "
            "core concepts without being overwhelmed."
        ),
        "intermediate": (
            "Ready to dive deeper! You can now upload your own data, configure "
            "train/test splits, use more algorithms, and build simple pipelines. "
            "Cross-validation and more metrics are available."
        ),
        "advanced": (
            "Full power unlocked! Access all algorithms, multivariate data handling, "
            "benchmarking tools, code export, and advanced configurations. "
            "You're ready for production workflows."
        ),
    }
    return descriptions.get(level, "")

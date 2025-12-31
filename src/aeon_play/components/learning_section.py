"""
Learning section component for educational content.
"""

from typing import Optional
from dash import html
import dash_bootstrap_components as dbc


def create_learning_section(
    title: str,
    content: str,
    level: str = "beginner",
    collapsed: bool = False,
    tips: Optional[list] = None,
) -> dbc.Card:
    """
    Create a learning section component.

    Args:
        title: Section title
        content: Main content text
        level: User level (beginner, intermediate, advanced)
        collapsed: Whether to start collapsed
        tips: Optional list of tip strings

    Returns:
        A Dash Bootstrap Card component
    """
    level_colors = {
        "beginner": "success",
        "intermediate": "warning",
        "advanced": "danger",
    }

    level_icons = {
        "beginner": "fa-seedling",
        "intermediate": "fa-tree",
        "advanced": "fa-mountain",
    }

    color = level_colors.get(level, "info")
    icon = level_icons.get(level, "fa-info-circle")

    body_content = [
        html.P(content, className="mb-2"),
    ]

    if tips:
        body_content.append(
            html.Ul(
                [html.Li(tip, className="text-muted small") for tip in tips],
                className="mb-0 ps-3",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.I(className=f"fas {icon} me-2"),
                    html.Strong(title),
                ],
                className="d-flex align-items-center mb-2",
            ),
            html.Div(body_content),
        ],
        className="learning-section fade-in",
    )


def create_concept_card(
    title: str,
    description: str,
    example: Optional[str] = None,
    icon: str = "fa-lightbulb",
) -> dbc.Card:
    """
    Create a concept explanation card.

    Args:
        title: Concept title
        description: Explanation text
        example: Optional code example
        icon: FontAwesome icon class

    Returns:
        A Dash Bootstrap Card
    """
    body = [
        html.H6(
            [html.I(className=f"fas {icon} me-2 text-warning"), title],
            className="card-title",
        ),
        html.P(description, className="card-text text-muted small"),
    ]

    if example:
        body.append(
            html.Pre(
                html.Code(example),
                className="bg-light p-2 rounded small",
                style={"fontSize": "0.8rem"},
            )
        )

    return dbc.Card(
        dbc.CardBody(body),
        className="mb-2 border-0 bg-light",
    )


def create_tip_alert(
    message: str,
    variant: str = "info",
    dismissable: bool = True,
) -> dbc.Alert:
    """
    Create a tip/info alert.

    Args:
        message: Alert message
        variant: Alert color (info, success, warning, danger)
        dismissable: Whether the alert can be dismissed

    Returns:
        A Dash Bootstrap Alert
    """
    icons = {
        "info": "fa-info-circle",
        "success": "fa-check-circle",
        "warning": "fa-exclamation-triangle",
        "danger": "fa-times-circle",
    }

    return dbc.Alert(
        [
            html.I(className=f"fas {icons.get(variant, 'fa-info-circle')} me-2"),
            message,
        ],
        color=variant,
        dismissable=dismissable,
        className="mb-3",
    )


def create_level_badge(level: str) -> html.Span:
    """Create a level indicator badge."""
    return html.Span(
        level.title(),
        className=f"level-badge level-{level}",
    )

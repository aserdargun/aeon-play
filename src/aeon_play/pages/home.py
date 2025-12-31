"""
Landing page for aeon-play.

The main entry point showcasing the platform's capabilities
and guiding users to choose their learning level.
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from aeon_play.components.level_gate import get_level_description

dash.register_page(__name__, path="/", name="Home", title="aeon-play | Home")


def create_hero_section():
    """Create the hero section."""
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H1(
                                        [
                                            html.I(className="fas fa-chart-line me-3"),
                                            "aeon-play",
                                        ],
                                        className="display-4 fw-bold mb-3",
                                    ),
                                    html.P(
                                        "Interactive Time Series Machine Learning Playground",
                                        className="lead mb-4",
                                    ),
                                    html.P(
                                        [
                                            "Explore, learn, and experiment with time series analysis ",
                                            "using the ",
                                            html.A(
                                                "aeon",
                                                href="https://www.aeon-toolkit.org/",
                                                target="_blank",
                                                className="text-white text-decoration-underline",
                                            ),
                                            " library - a scikit-learn compatible toolkit for ",
                                            "time series machine learning.",
                                        ],
                                        className="mb-4 opacity-90",
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                [
                                                    html.I(className="fas fa-play me-2"),
                                                    "Get Started",
                                                ],
                                                href="/data",
                                                color="light",
                                                size="lg",
                                                className="me-3",
                                            ),
                                            dbc.Button(
                                                [
                                                    html.I(className="fas fa-book me-2"),
                                                    "Learn More",
                                                ],
                                                href="https://www.aeon-toolkit.org/en/stable/",
                                                target="_blank",
                                                color="outline-light",
                                                size="lg",
                                            ),
                                        ],
                                    ),
                                ],
                                lg=8,
                            ),
                        ],
                        justify="center",
                        className="text-center",
                    ),
                ],
            ),
        ],
        className="hero-section text-center",
    )


def create_level_selector():
    """Create the learning level selector section."""
    levels = [
        {
            "id": "beginner",
            "name": "Beginner",
            "icon": "fa-seedling",
            "color": "success",
            "features": [
                "Curated datasets",
                "Simple algorithms",
                "Guided workflows",
                "Basic visualizations",
            ],
        },
        {
            "id": "intermediate",
            "name": "Intermediate",
            "icon": "fa-tree",
            "color": "warning",
            "features": [
                "Custom data upload",
                "More algorithms",
                "Cross-validation",
                "Pipeline building",
            ],
        },
        {
            "id": "advanced",
            "name": "Advanced",
            "icon": "fa-mountain",
            "color": "danger",
            "features": [
                "All algorithms",
                "Multivariate data",
                "Benchmarking",
                "Code export",
            ],
        },
    ]

    cards = []
    for level in levels:
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.I(className=f"fas {level['icon']} fa-2x mb-2"),
                                html.H5(level["name"], className="mb-0"),
                            ],
                            className=f"text-center bg-{level['color']} text-white",
                        ),
                        dbc.CardBody(
                            [
                                html.P(
                                    get_level_description(level["id"]),
                                    className="text-muted small mb-3",
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            [html.I(className="fas fa-check text-success me-2"), f],
                                            className="small",
                                        )
                                        for f in level["features"]
                                    ],
                                    className="list-unstyled mb-3",
                                ),
                                dbc.Button(
                                    f"Start as {level['name']}",
                                    id=f"home-select-{level['id']}",
                                    color=level["color"],
                                    className="w-100",
                                ),
                            ],
                        ),
                    ],
                    className="h-100",
                ),
                md=4,
                className="mb-4",
            )
        )

    return html.Div(
        [
            html.H2("Choose Your Learning Level", className="text-center mb-4"),
            html.P(
                "Select a level that matches your experience. You can change this anytime.",
                className="text-center text-muted mb-4",
            ),
            dbc.Row(cards),
        ],
        className="py-5",
    )


def create_task_cards():
    """Create task overview cards."""
    tasks = [
        {
            "name": "Classification",
            "icon": "fa-tags",
            "desc": "Assign time series to categories",
            "href": "/tasks/classification",
            "class": "classification",
        },
        {
            "name": "Regression",
            "icon": "fa-chart-line",
            "desc": "Predict continuous values from series",
            "href": "/tasks/regression",
            "class": "regression",
        },
        {
            "name": "Clustering",
            "icon": "fa-object-group",
            "desc": "Group similar time series together",
            "href": "/tasks/clustering",
            "class": "clustering",
        },
        {
            "name": "Forecasting",
            "icon": "fa-arrow-trend-up",
            "desc": "Predict future values of a series",
            "href": "/tasks/forecasting",
            "class": "forecasting",
        },
        {
            "name": "Anomaly Detection",
            "icon": "fa-exclamation-triangle",
            "desc": "Find unusual patterns in series",
            "href": "/tasks/anomaly",
            "class": "anomaly",
        },
        {
            "name": "Segmentation",
            "icon": "fa-scissors",
            "desc": "Find change points in series",
            "href": "/tasks/segmentation",
            "class": "segmentation",
        },
    ]

    cards = []
    for task in tasks:
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.I(className=f"fas {task['icon']} me-2"),
                                task["name"],
                            ],
                        ),
                        dbc.CardBody(
                            [
                                html.P(task["desc"], className="card-text small"),
                                dbc.Button(
                                    "Explore",
                                    href=task["href"],
                                    color="primary",
                                    size="sm",
                                    outline=True,
                                ),
                            ],
                        ),
                    ],
                    className=f"feature-card {task['class']}",
                ),
                md=4,
                lg=2,
                className="mb-3",
            )
        )

    return html.Div(
        [
            html.H2("Explore Time Series Tasks", className="text-center mb-4"),
            html.P(
                "aeon supports a comprehensive set of time series machine learning tasks.",
                className="text-center text-muted mb-4",
            ),
            dbc.Row(cards, className="g-3"),
        ],
        className="py-5 bg-light rounded",
    )


def create_features_section():
    """Create features overview section."""
    features = [
        {
            "icon": "fa-database",
            "title": "Data Studio",
            "desc": "Load built-in datasets or upload your own CSV/TS files. Preview, validate, and prepare data for analysis.",
            "href": "/data",
        },
        {
            "icon": "fa-eye",
            "title": "Visual Lab",
            "desc": "Explore time series with interactive visualizations. View correlations, spectrograms, and patterns.",
            "href": "/viz",
        },
        {
            "icon": "fa-wand-magic-sparkles",
            "title": "Transformations",
            "desc": "Apply feature extraction and preprocessing. Explore Catch22, wavelets, shapelets, and more.",
            "href": "/transform",
        },
        {
            "icon": "fa-ruler",
            "title": "Distances",
            "desc": "Learn about time series distance measures. Compare DTW, Euclidean, and edit-based distances.",
            "href": "/distances",
        },
        {
            "icon": "fa-scale-balanced",
            "title": "Benchmarking",
            "desc": "Compare multiple algorithms on your data. Export results and generate comparison reports.",
            "href": "/benchmark",
        },
        {
            "icon": "fa-code",
            "title": "Code Export",
            "desc": "Generate reproducible Python scripts. Take your experiments from playground to production.",
            "href": "/tasks/classification",
        },
    ]

    cards = []
    for f in features:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                html.I(className=f"fas {f['icon']} fa-2x text-primary"),
                                className="mb-3",
                            ),
                            html.H5(f["title"], className="card-title"),
                            html.P(f["desc"], className="card-text small text-muted"),
                            dbc.Button(
                                "Go",
                                href=f["href"],
                                color="outline-primary",
                                size="sm",
                            ),
                        ],
                        className="text-center",
                    ),
                    className="h-100 border-0 shadow-sm",
                ),
                md=4,
                className="mb-4",
            )
        )

    return html.Div(
        [
            html.H2("Platform Features", className="text-center mb-4"),
            dbc.Row(cards),
        ],
        className="py-5",
    )


def create_aeon_info():
    """Create aeon library information section."""
    return html.Div(
        [
            html.H2("About aeon", className="text-center mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                [
                                    "aeon is an open-source Python library for time series machine learning. ",
                                    "It provides a unified, scikit-learn compatible interface for:",
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Ul(
                                            [
                                                html.Li("Time series classification"),
                                                html.Li("Time series regression"),
                                                html.Li("Time series clustering"),
                                            ],
                                        ),
                                    ),
                                    dbc.Col(
                                        html.Ul(
                                            [
                                                html.Li("Forecasting"),
                                                html.Li("Anomaly detection"),
                                                html.Li("Segmentation"),
                                            ],
                                        ),
                                    ),
                                ],
                                className="mb-3",
                            ),
                            html.P(
                                [
                                    "Learn more at ",
                                    html.A(
                                        "aeon-toolkit.org",
                                        href="https://www.aeon-toolkit.org/",
                                        target="_blank",
                                    ),
                                    " or check the ",
                                    html.A(
                                        "documentation",
                                        href="https://www.aeon-toolkit.org/en/stable/",
                                        target="_blank",
                                    ),
                                    ".",
                                ],
                            ),
                        ],
                        md=8,
                    ),
                ],
                justify="center",
            ),
        ],
        className="py-5 bg-light rounded text-center",
    )


# Page layout
layout = html.Div(
    [
        create_hero_section(),
        dbc.Container(
            [
                create_level_selector(),
                html.Hr(),
                create_task_cards(),
                html.Hr(),
                create_features_section(),
                html.Hr(),
                create_aeon_info(),
            ],
            fluid=True,
        ),
    ],
)


# Callbacks for level selection
@callback(
    Output("user-level", "data", allow_duplicate=True),
    Input("home-select-beginner", "n_clicks"),
    Input("home-select-intermediate", "n_clicks"),
    Input("home-select-advanced", "n_clicks"),
    prevent_initial_call=True,
)
def select_level(n_beg, n_int, n_adv):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "beginner" in button_id:
        return "beginner"
    elif "intermediate" in button_id:
        return "intermediate"
    elif "advanced" in button_id:
        return "advanced"

    return dash.no_update

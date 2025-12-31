"""
Main Dash application entry point for aeon-play.

This module initializes the multi-page Dash application with:
- Bootstrap theming
- Caching configuration
- Background callback support
- Level-based progressive disclosure
"""

import os
from pathlib import Path

import dash
from dash import Dash, html, dcc, callback, Input, Output, State, clientside_callback
import dash_bootstrap_components as dbc
from flask_caching import Cache
import diskcache

# Configuration from environment
DEBUG = os.getenv("AEON_PLAY_DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8050"))
HOST = os.getenv("HOST", "0.0.0.0")
CACHE_DIR = os.getenv("AEON_PLAY_CACHE_DIR", "./.cache")

# Ensure cache directory exists
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Initialize diskcache for background callbacks
cache_disk = diskcache.Cache(CACHE_DIR)
background_callback_manager = dash.DiskcacheManager(cache_disk)

# Initialize Dash app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.FONT_AWESOME,
    ],
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "aeon-play: Interactive Time Series Machine Learning Playground - Learn classification, forecasting, clustering, and more with the aeon library."},
        {"name": "keywords", "content": "time series, machine learning, aeon, classification, forecasting, clustering, anomaly detection"},
        {"property": "og:title", "content": "aeon-play: Time Series ML Playground"},
        {"property": "og:description", "content": "Interactive educational platform for time series machine learning"},
        {"property": "og:type", "content": "website"},
        {"property": "og:url", "content": "https://aeon-play.org"},
    ],
    title="aeon-play | Time Series ML Playground",
)

# Flask-Caching for computational results
flask_cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": os.path.join(CACHE_DIR, "flask_cache"),
        "CACHE_DEFAULT_TIMEOUT": 3600,
    },
)

# Server reference for gunicorn
server = app.server


def create_navbar():
    """Create the navigation bar with level selector."""
    return dbc.Navbar(
        dbc.Container(
            [
                # Brand
                dbc.NavbarBrand(
                    [
                        html.I(className="fas fa-chart-line me-2"),
                        "aeon-play",
                    ],
                    href="/",
                    className="fw-bold",
                ),
                # Toggle for mobile
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                # Collapsible content
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Data Studio", href="/data", id="nav-data")),
                            dbc.NavItem(dbc.NavLink("Visual Lab", href="/viz", id="nav-viz")),
                            dbc.NavItem(dbc.NavLink("Transforms", href="/transform", id="nav-transform")),
                            dbc.NavItem(dbc.NavLink("Distances", href="/distances", id="nav-distances")),
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem("Classification", href="/tasks/classification"),
                                    dbc.DropdownMenuItem("Regression", href="/tasks/regression"),
                                    dbc.DropdownMenuItem("Clustering", href="/tasks/clustering"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem("Forecasting", href="/tasks/forecasting"),
                                    dbc.DropdownMenuItem("Anomaly Detection", href="/tasks/anomaly"),
                                    dbc.DropdownMenuItem("Segmentation", href="/tasks/segmentation"),
                                    dbc.DropdownMenuItem("Similarity Search", href="/tasks/similarity"),
                                ],
                                nav=True,
                                in_navbar=True,
                                label="Tasks",
                                id="nav-tasks-dropdown",
                            ),
                            dbc.NavItem(dbc.NavLink("Benchmark", href="/benchmark", id="nav-benchmark")),
                            # Level selector
                            html.Div(
                                [
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Beginner",
                                                id="btn-level-beginner",
                                                color="success",
                                                outline=True,
                                                size="sm",
                                                className="level-btn",
                                            ),
                                            dbc.Button(
                                                "Intermediate",
                                                id="btn-level-intermediate",
                                                color="warning",
                                                outline=True,
                                                size="sm",
                                                className="level-btn",
                                            ),
                                            dbc.Button(
                                                "Advanced",
                                                id="btn-level-advanced",
                                                color="danger",
                                                outline=True,
                                                size="sm",
                                                className="level-btn",
                                            ),
                                        ],
                                        size="sm",
                                    ),
                                ],
                                className="ms-auto d-flex align-items-center",
                            ),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        className="mb-3",
        sticky="top",
    )


def create_footer():
    """Create the footer."""
    return html.Footer(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        html.P(
                            [
                                "Powered by ",
                                html.A("aeon", href="https://www.aeon-toolkit.org/", target="_blank"),
                                " | ",
                                html.A("Documentation", href="https://www.aeon-toolkit.org/en/stable/", target="_blank"),
                                " | ",
                                html.A("GitHub", href="https://github.com/aeon-toolkit/aeon", target="_blank"),
                            ],
                            className="text-muted mb-0",
                        ),
                        width=12,
                        className="text-center py-3",
                    ),
                ],
            ),
            fluid=True,
        ),
        className="bg-light mt-5 border-top",
    )


# Main layout
app.layout = html.Div(
    [
        # URL and location tracking
        dcc.Location(id="url", refresh=False),

        # Global stores
        dcc.Store(id="user-level", storage_type="local", data="beginner"),
        dcc.Store(id="session-datasets", storage_type="session", data={}),
        dcc.Store(id="current-dataset-key", storage_type="session", data=None),

        # Navigation
        create_navbar(),

        # Page content
        dbc.Container(
            dash.page_container,
            fluid=True,
            className="main-content px-4",
        ),

        # Footer
        create_footer(),

        # Toast notifications container
        html.Div(id="toast-container", className="position-fixed bottom-0 end-0 p-3"),
    ],
    className="d-flex flex-column min-vh-100",
)


# Navbar toggle callback
@callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
)
def toggle_navbar_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Level selector callbacks
@callback(
    Output("user-level", "data"),
    Output("btn-level-beginner", "outline"),
    Output("btn-level-intermediate", "outline"),
    Output("btn-level-advanced", "outline"),
    Input("btn-level-beginner", "n_clicks"),
    Input("btn-level-intermediate", "n_clicks"),
    Input("btn-level-advanced", "n_clicks"),
    State("user-level", "data"),
    prevent_initial_call=True,
)
def update_level(n_beg, n_int, n_adv, current_level):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, True, True, True

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "btn-level-beginner":
        return "beginner", False, True, True
    elif button_id == "btn-level-intermediate":
        return "intermediate", True, False, True
    elif button_id == "btn-level-advanced":
        return "advanced", True, True, False

    return current_level, True, True, True


# Initial level button state
@callback(
    Output("btn-level-beginner", "outline", allow_duplicate=True),
    Output("btn-level-intermediate", "outline", allow_duplicate=True),
    Output("btn-level-advanced", "outline", allow_duplicate=True),
    Input("user-level", "data"),
    prevent_initial_call="initial_duplicate",
)
def set_initial_level_buttons(level):
    if level == "beginner":
        return False, True, True
    elif level == "intermediate":
        return True, False, True
    elif level == "advanced":
        return True, True, False
    return True, True, True


def main():
    """Run the application."""
    app.run(debug=DEBUG, host=HOST, port=PORT)


if __name__ == "__main__":
    main()

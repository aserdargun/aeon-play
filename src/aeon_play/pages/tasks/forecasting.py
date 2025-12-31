"""
Forecasting task module for time series forecasting.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

from aeon_play.services.aeon_discovery import get_aeon_discovery
from aeon_play.services.runners import TaskRunner
from aeon_play.components.learning_section import create_learning_section
from aeon_play.components.metrics_display import create_metrics_display, create_run_logs

dash.register_page(
    __name__,
    path="/tasks/forecasting",
    name="Forecasting",
    title="aeon-play | Forecasting",
)


def create_forecast_plot(
    historical: np.ndarray,
    forecast: np.ndarray,
    actual: np.ndarray = None,
) -> go.Figure:
    """Create forecast visualization."""
    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            y=historical,
            mode='lines',
            name='Historical',
            line=dict(color='#2c3e50', width=2),
        )
    )

    # Forecast
    forecast_x = list(range(len(historical), len(historical) + len(forecast)))
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#e74c3c', width=2, dash='dash'),
        )
    )

    # Actual if provided
    if actual is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color='#2ecc71', width=2),
            )
        )

    fig.update_layout(
        title="Forecast Results",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


layout = html.Div(
    [
        create_learning_section(
            title="Time Series Forecasting",
            content=(
                "Forecasting predicts future values of a time series. "
                "Note: Forecasting in aeon is experimental. For production, "
                "consider sktime or other dedicated forecasting libraries."
            ),
            tips=[
                "Start with NaiveForecaster as a baseline",
                "Horizon is how many steps ahead to predict",
                "Backtest on held-out data to evaluate accuracy",
            ],
        ),

        dbc.Alert(
            [
                html.I(className="fas fa-flask me-2"),
                "Forecasting in aeon is currently experimental. "
                "Features may change in future versions.",
            ],
            color="warning",
            className="mb-3",
        ),

        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Select or Generate Series"),
                                        dbc.Select(
                                            id="fc-series-type",
                                            options=[
                                                {"label": "Airline Passengers", "value": "airline"},
                                                {"label": "Sine + Trend", "value": "sine_trend"},
                                                {"label": "Random Walk", "value": "random_walk"},
                                                {"label": "Seasonal", "value": "seasonal"},
                                            ],
                                            value="airline",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Forecaster"),
                                dbc.CardBody(
                                    [
                                        dbc.Select(
                                            id="fc-estimator-select",
                                            options=[],
                                            className="mb-2",
                                        ),
                                        dbc.Label("Forecast Horizon", className="small"),
                                        dbc.Input(
                                            id="fc-horizon",
                                            type="number",
                                            min=1, max=50, value=10,
                                            size="sm",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Run Forecast"],
                            id="btn-fc-run",
                            color="primary",
                            className="w-100",
                        ),
                    ],
                    md=3,
                ),

                # Main
                dbc.Col(
                    [
                        dcc.Loading(
                            [
                                dbc.Alert(id="fc-status-alert", is_open=False, dismissable=True),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Series Preview"),
                                        dbc.CardBody(
                                            dcc.Graph(id="fc-series-plot", config={"displayModeBar": False})
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Forecast Result"),
                                        dbc.CardBody(
                                            dcc.Graph(id="fc-forecast-plot", config={"displayModeBar": False})
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
                                dbc.CardHeader("Metrics"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="fc-metrics-container"),
                                        html.Div(id="fc-logs-container", className="mt-3"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
        ),

        dcc.Store(id="fc-series-data"),
    ],
)


@callback(
    Output("fc-estimator-select", "options"),
    Input("user-level", "data"),
)
def update_forecasters(level):
    discovery = get_aeon_discovery()
    forecasters = discovery.discover_estimators("forecaster", level or "beginner")

    options = [{"label": f.name, "value": f.name} for f in forecasters]
    if not options:
        options = [
            {"label": "NaiveForecaster", "value": "NaiveForecaster"},
            {"label": "TrendForecaster", "value": "TrendForecaster"},
        ]
    return options


@callback(
    Output("fc-series-data", "data"),
    Output("fc-series-plot", "figure"),
    Input("fc-series-type", "value"),
)
def load_series(series_type):
    np.random.seed(42)

    if series_type == "airline":
        t = np.arange(144)
        trend = t * 2.5
        seasonal = 50 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 10, 144)
        series = 100 + trend + seasonal + noise

    elif series_type == "sine_trend":
        t = np.linspace(0, 10, 200)
        series = t * 5 + 20 * np.sin(2 * np.pi * t / 2) + np.random.randn(200) * 3

    elif series_type == "random_walk":
        series = np.cumsum(np.random.randn(150))

    else:  # seasonal
        t = np.arange(120)
        series = 100 + 30 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 5

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, mode='lines', line=dict(color='#2c3e50')))
    fig.update_layout(
        title=f"Time Series: {series_type}",
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return series.tolist(), fig


@callback(
    Output("fc-status-alert", "is_open"),
    Output("fc-status-alert", "children"),
    Output("fc-status-alert", "color"),
    Output("fc-forecast-plot", "figure"),
    Output("fc-metrics-container", "children"),
    Output("fc-logs-container", "children"),
    Input("btn-fc-run", "n_clicks"),
    State("fc-series-data", "data"),
    State("fc-estimator-select", "value"),
    State("fc-horizon", "value"),
    prevent_initial_call=True,
)
def run_forecast(n_clicks, series_data, estimator, horizon):
    if series_data is None:
        return True, "Load series first", "warning", go.Figure(), None, None

    series = np.array(series_data)
    horizon = horizon or 10

    runner = TaskRunner()
    result = runner.run_forecasting(
        series=series.reshape(1, 1, -1),
        estimator_name=estimator or "NaiveForecaster",
        params={},
        horizon=horizon,
    )

    if result.success:
        forecast = np.array(result.forecast_values)
        train_len = len(series) - horizon
        historical = series[:train_len]
        actual = series[train_len:]

        fig = create_forecast_plot(historical, forecast, actual)
        metrics = create_metrics_display(result.metrics, "forecasting")
        logs = create_run_logs(result.logs)

        return (
            True, f"Completed in {result.train_time:.2f}s", "success",
            fig, metrics, logs,
        )
    else:
        return (
            True, result.error_message, "danger",
            go.Figure(), None, create_run_logs(result.logs) if result.logs else None,
        )

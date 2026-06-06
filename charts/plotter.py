"""Interactive Plotly chart generation."""

from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forecasting.linear_forecast import ForecastResult
from utils.constants import METRIC_DISPLAY_NAMES


def create_forecast_chart(
    forecast: ForecastResult,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create an interactive Plotly chart with historical data, forecast, and threshold.
    """
    display_name = METRIC_DISPLAY_NAMES.get(forecast.metric, forecast.metric)
    chart_title = title or f"{display_name} — Forecast ({forecast.model_type.upper()})"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast.historical_dates,
            y=forecast.historical_values,
            mode="lines+markers",
            name="Historical",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6),
        )
    )

    if forecast.forecast_dates and forecast.forecast_values:
        connector_x = [forecast.historical_dates[-1], forecast.forecast_dates[0]]
        connector_y = [forecast.historical_values[-1], forecast.forecast_values[0]]

        fig.add_trace(
            go.Scatter(
                x=connector_x,
                y=connector_y,
                mode="lines",
                line=dict(color="#16a34a", width=2, dash="dot"),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast.forecast_dates,
                y=forecast.forecast_values,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#16a34a", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
            )
        )

    all_dates = list(forecast.historical_dates) + list(forecast.forecast_dates or [])
    if all_dates:
        fig.add_trace(
            go.Scatter(
                x=[all_dates[0], all_dates[-1]],
                y=[forecast.threshold, forecast.threshold],
                mode="lines",
                name=f"Threshold ({forecast.threshold:.0f}%)",
                line=dict(color="#dc2626", width=2, dash="dashdot"),
            )
        )

    if forecast.threshold_crossing_date:
        crossing_value = forecast.threshold
        fig.add_annotation(
            x=forecast.threshold_crossing_date,
            y=crossing_value,
            text=f"Crosses {forecast.threshold:.0f}%",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            font=dict(size=11, color="#dc2626"),
        )
        fig.add_trace(
            go.Scatter(
                x=[forecast.threshold_crossing_date],
                y=[crossing_value],
                mode="markers",
                name="Threshold Crossing",
                marker=dict(color="#dc2626", size=12, symbol="x"),
            )
        )

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Utilization (%)",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    return fig


def create_multi_metric_chart(
    forecasts: dict[str, ForecastResult],
    title: str = "Capacity Forecast — All Resources",
) -> go.Figure:
    """Create a comparison chart for multiple metrics."""
    colors = {"cpu_usage": "#2563eb", "memory_usage": "#7c3aed", "disk_usage": "#ea580c"}
    fig = make_subplots(rows=1, cols=1)

    for metric, forecast in forecasts.items():
        color = colors.get(metric, "#64748b")
        display = METRIC_DISPLAY_NAMES.get(metric, metric)

        fig.add_trace(
            go.Scatter(
                x=forecast.historical_dates,
                y=forecast.historical_values,
                mode="lines",
                name=f"{display} (Historical)",
                line=dict(color=color, width=2),
            )
        )
        if forecast.forecast_dates:
            fig.add_trace(
                go.Scatter(
                    x=forecast.forecast_dates,
                    y=forecast.forecast_values,
                    mode="lines",
                    name=f"{display} (Forecast)",
                    line=dict(color=color, width=2, dash="dash"),
                )
            )

    threshold = next(iter(forecasts.values())).threshold if forecasts else 80.0
    first_fc = next(iter(forecasts.values()))
    all_dates = list(first_fc.historical_dates) + list(first_fc.forecast_dates or [])

    fig.add_trace(
        go.Scatter(
            x=[all_dates[0], all_dates[-1]],
            y=[threshold, threshold],
            mode="lines",
            name=f"Threshold ({threshold:.0f}%)",
            line=dict(color="#dc2626", width=2, dash="dashdot"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Utilization (%)",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

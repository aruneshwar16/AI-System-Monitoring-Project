"""ARIMA forecasting engine."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from forecasting.linear_forecast import ForecastResult, _find_threshold_crossing
from utils.constants import SUPPORTED_METRICS
from utils.logger import get_logger

logger = get_logger(__name__)

ARIMA_ORDER = (1, 1, 1)


def forecast_metric_arima(
    df: pd.DataFrame,
    metric: str,
    forecast_months: int = 6,
    threshold: float = 80.0,
    growth_multiplier: float = 1.0,
) -> ForecastResult:
    """
    Forecast a single metric using ARIMA(1,1,1).

    Falls back to linear trend slope for threshold crossing when ARIMA
    produces flat or declining forecasts with growth_multiplier != 1.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")

    dates = df["date"]
    values = df[metric].values.astype(float)

    try:
        model = ARIMA(values, order=ARIMA_ORDER)
        fitted = model.fit()
        forecast_values = fitted.forecast(steps=forecast_months)
    except Exception as exc:
        logger.warning("ARIMA fit failed for %s: %s. Using last-value projection.", metric, exc)
        last_val = values[-1]
        forecast_values = np.full(forecast_months, last_val)

    if growth_multiplier != 1.0:
        baseline_slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
        adjustment = baseline_slope * (growth_multiplier - 1.0)
        for i in range(forecast_months):
            forecast_values[i] += adjustment * (i + 1)

    forecast_values = np.clip(forecast_values, 0, 100)

    last_date = dates.max()
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS",
    )

    origin = dates.min()
    months = (dates - origin).dt.days / 30.44
    slope = float(np.polyfit(months, values, 1)[0]) * growth_multiplier
    intercept = float(values.mean() - slope * months.mean())

    y_mean = np.mean(values)
    ss_tot = np.sum((values - y_mean) ** 2)
    try:
        fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
        if len(fitted_values) > len(values):
            fitted_values = fitted_values[-len(values) :]
        ss_res = np.sum((values - fitted_values) ** 2)
        r_squared = float(max(0, 1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
    except Exception:
        r_squared = 0.5

    threshold_crossing = _find_arima_threshold_crossing(
        last_date=last_date,
        forecast_dates=forecast_dates,
        forecast_values=forecast_values,
        threshold=threshold,
        origin=origin,
        slope=slope,
        intercept=intercept,
    )

    return ForecastResult(
        metric=metric,
        historical_dates=dates.dt.to_pydatetime().tolist(),
        historical_values=values.tolist(),
        forecast_dates=forecast_dates.to_pydatetime().tolist(),
        forecast_values=forecast_values.tolist(),
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        threshold=threshold,
        threshold_crossing_date=threshold_crossing,
        model_type="arima",
    )


def _find_arima_threshold_crossing(
    last_date: pd.Timestamp,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    threshold: float,
    origin: pd.Timestamp,
    slope: float,
    intercept: float,
) -> Optional[datetime]:
    """Detect threshold crossing in ARIMA forecast series."""
    current = forecast_values[0] if len(forecast_values) else 0
    if current >= threshold:
        return (last_date + pd.DateOffset(months=1)).to_pydatetime()

    for i, val in enumerate(forecast_values):
        if val >= threshold:
            return forecast_dates[i].to_pydatetime()

    return _find_threshold_crossing(
        origin=origin,
        slope=slope,
        intercept=intercept,
        threshold=threshold,
        start_date=last_date,
        max_months=len(forecast_values) * 4,
    )

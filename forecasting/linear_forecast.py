"""Linear regression forecasting engine."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.constants import SUPPORTED_METRICS
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast output."""

    metric: str
    historical_dates: list[datetime]
    historical_values: list[float]
    forecast_dates: list[datetime]
    forecast_values: list[float]
    slope: float
    intercept: float
    r_squared: float
    threshold: float
    threshold_crossing_date: Optional[datetime]
    model_type: str = "linear"


def _prepare_features(dates: pd.Series) -> np.ndarray:
    """Convert dates to numeric month offsets from first date."""
    origin = dates.min()
    months = (dates - origin).dt.days / 30.44
    return months.values.reshape(-1, 1)


def forecast_metric_linear(
    df: pd.DataFrame,
    metric: str,
    forecast_months: int = 6,
    threshold: float = 80.0,
    growth_multiplier: float = 1.0,
) -> ForecastResult:
    """
    Forecast a single metric using linear regression.

    Args:
        df: DataFrame with 'date' and metric columns.
        metric: Metric column name.
        forecast_months: Number of months to forecast ahead.
        threshold: Threshold percentage for crossing detection.
        growth_multiplier: Scales the slope for what-if scenarios.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")

    dates = df["date"]
    values = df[metric].values.astype(float)
    X = _prepare_features(dates)
    y = values

    model = LinearRegression()
    model.fit(X, y)

    slope = float(model.coef_[0]) * growth_multiplier
    intercept = float(model.intercept_)

    y_pred_hist = model.predict(X)
    ss_res = np.sum((y - y_pred_hist) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    last_date = dates.max()
    origin = dates.min()
    last_month_offset = (last_date - origin).days / 30.44

    forecast_offsets = np.array(
        [last_month_offset + i + 1 for i in range(forecast_months)]
    ).reshape(-1, 1)
    forecast_values = (slope * forecast_offsets.flatten()) + intercept
    forecast_values = np.clip(forecast_values, 0, 100)

    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS",
    )

    threshold_crossing = _find_threshold_crossing(
        origin=origin,
        slope=slope,
        intercept=intercept,
        threshold=threshold,
        start_date=last_date,
        max_months=forecast_months * 4,
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
        model_type="linear",
    )


def _find_threshold_crossing(
    origin: pd.Timestamp,
    slope: float,
    intercept: float,
    threshold: float,
    start_date: pd.Timestamp,
    max_months: int = 48,
) -> Optional[datetime]:
    """Find when the linear trend crosses the threshold."""
    if slope <= 0:
        current_value = slope * ((start_date - origin).days / 30.44) + intercept
        if current_value >= threshold:
            return start_date.to_pydatetime()
        return None

    for month in range(1, max_months + 1):
        offset = (start_date - origin).days / 30.44 + month
        projected = slope * offset + intercept
        if projected >= threshold:
            crossing_date = start_date + pd.DateOffset(months=month)
            return crossing_date.to_pydatetime()
    return None


def compare_metrics_threshold_crossing(
    df: pd.DataFrame,
    forecast_months: int = 12,
    threshold: float = 80.0,
    growth_multipliers: Optional[dict[str, float]] = None,
) -> list[tuple[str, Optional[datetime], float]]:
    """
    Compare all metrics and return ordered list of threshold crossings.

    Returns list of (metric, crossing_date, current_value) sorted by crossing date.
    """
    growth_multipliers = growth_multipliers or {}
    results = []

    for metric in SUPPORTED_METRICS:
        multiplier = growth_multipliers.get(metric, 1.0)
        forecast = forecast_metric_linear(
            df, metric, forecast_months, threshold, multiplier
        )
        current_value = df[metric].iloc[-1]
        results.append((metric, forecast.threshold_crossing_date, float(current_value)))

    with_date = [(m, d, v) for m, d, v in results if d is not None]
    without_date = [(m, d, v) for m, d, v in results if d is None]

    with_date.sort(key=lambda x: x[1])
    return with_date + without_date

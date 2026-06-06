"""Confidence score calculation for forecasts."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from forecasting.linear_forecast import ForecastResult
from utils.constants import RiskLevel


@dataclass
class ConfidenceScore:
    """Composite confidence score breakdown."""

    overall: float
    model_fit: float
    data_quality: float
    forecast_horizon: float
    parse_confidence: float
    validation_adjustment: float
    label: str


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "Very High"
    if score >= 0.70:
        return "High"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.40:
        return "Low"
    return "Very Low"


def calculate_data_quality_score(data_points: int, missing_ratio: float = 0.0) -> float:
    """Score based on sample size and completeness."""
    size_score = min(1.0, data_points / 24.0)
    completeness = 1.0 - missing_ratio
    return round(0.6 * size_score + 0.4 * completeness, 3)


def calculate_horizon_penalty(forecast_months: int, max_months: int = 24) -> float:
    """Shorter horizons yield higher confidence."""
    penalty = forecast_months / max_months
    return round(max(0.3, 1.0 - 0.5 * penalty), 3)


def calculate_confidence(
    forecast: ForecastResult,
    data_points: int,
    forecast_months: int,
    parse_confidence: float = 0.85,
    validation_adjustment: float = 1.0,
) -> ConfidenceScore:
    """
    Compute composite confidence score.

    Components:
    - model_fit: R-squared from regression / ARIMA fit quality
    - data_quality: sample size and completeness
    - forecast_horizon: penalty for longer forecasts
    - parse_confidence: NL parser certainty
    - validation_adjustment: agent validation step multiplier
    """
    model_fit = max(0.0, min(1.0, forecast.r_squared))
    if forecast.model_type == "arima":
        model_fit = max(model_fit, 0.45)

    data_quality = calculate_data_quality_score(data_points)
    horizon = calculate_horizon_penalty(forecast_months)
    parse_conf = max(0.0, min(1.0, parse_confidence))
    validation = max(0.0, min(1.0, validation_adjustment))

    overall = (
        0.30 * model_fit
        + 0.25 * data_quality
        + 0.20 * horizon
        + 0.15 * parse_conf
        + 0.10 * validation
    )
    overall = round(max(0.0, min(1.0, overall)), 3)

    return ConfidenceScore(
        overall=overall,
        model_fit=round(model_fit, 3),
        data_quality=round(data_quality, 3),
        forecast_horizon=round(horizon, 3),
        parse_confidence=round(parse_conf, 3),
        validation_adjustment=round(validation, 3),
        label=_score_label(overall),
    )


def risk_from_utilization(value: float) -> RiskLevel:
    """Map utilization percentage to risk level."""
    from utils.constants import RISK_HIGH_MAX, RISK_LOW_MAX, RISK_MEDIUM_MAX

    if value >= RISK_HIGH_MAX:
        return RiskLevel.CRITICAL
    if value >= RISK_MEDIUM_MAX:
        return RiskLevel.HIGH
    if value >= RISK_LOW_MAX:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def days_until(date_from, target_date) -> Optional[int]:
    """Calculate days between two dates."""
    if target_date is None:
        return None
    delta = target_date - date_from
    return max(0, delta.days)

"""Risk analysis for infrastructure capacity metrics."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from forecasting.confidence import days_until, risk_from_utilization
from forecasting.linear_forecast import ForecastResult
from utils.constants import METRIC_DISPLAY_NAMES, RiskLevel, SUPPORTED_METRICS


@dataclass
class MetricRisk:
    """Risk assessment for a single metric."""

    metric: str
    display_name: str
    current_value: float
    projected_end_value: float
    risk_level: RiskLevel
    threshold: float
    threshold_crossing_date: Optional[datetime]
    days_to_threshold: Optional[int]
    trend_slope: float


@dataclass
class RiskReport:
    """Aggregated risk analysis across all metrics."""

    metrics: list[MetricRisk] = field(default_factory=list)
    highest_risk_metric: Optional[str] = None
    first_threshold_metric: Optional[str] = None
    first_threshold_date: Optional[datetime] = None
    overall_risk: RiskLevel = RiskLevel.LOW

    def to_dict(self) -> dict:
        return {
            "overall_risk": self.overall_risk.value,
            "highest_risk_metric": self.highest_risk_metric,
            "first_threshold_metric": self.first_threshold_metric,
            "first_threshold_date": (
                self.first_threshold_date.isoformat()
                if self.first_threshold_date
                else None
            ),
            "metrics": [
                {
                    "metric": m.metric,
                    "display_name": m.display_name,
                    "current_value": round(m.current_value, 2),
                    "projected_end_value": round(m.projected_end_value, 2),
                    "risk_level": m.risk_level.value,
                    "threshold": m.threshold,
                    "threshold_crossing_date": (
                        m.threshold_crossing_date.isoformat()
                        if m.threshold_crossing_date
                        else None
                    ),
                    "days_to_threshold": m.days_to_threshold,
                    "trend_slope": round(m.trend_slope, 4),
                }
                for m in self.metrics
            ],
        }


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class RiskAnalyzer:
    """Analyzes capacity risk from forecast results."""

    def analyze_single(
        self,
        df: pd.DataFrame,
        forecast: ForecastResult,
        reference_date: Optional[datetime] = None,
    ) -> MetricRisk:
        """Build risk profile for one metric."""
        current = float(df[forecast.metric].iloc[-1])
        projected_end = forecast.forecast_values[-1] if forecast.forecast_values else current
        ref = reference_date or df["date"].iloc[-1].to_pydatetime()

        crossing = forecast.threshold_crossing_date
        days = days_until(ref, crossing) if crossing else None

        current_risk = risk_from_utilization(current)
        projected_risk = risk_from_utilization(projected_end)
        risk_level = current_risk if _RISK_ORDER[current_risk] >= _RISK_ORDER[projected_risk] else projected_risk

        if projected_end >= forecast.threshold:
            risk_level = max([risk_level, RiskLevel.HIGH], key=lambda r: _RISK_ORDER[r])

        return MetricRisk(
            metric=forecast.metric,
            display_name=METRIC_DISPLAY_NAMES.get(forecast.metric, forecast.metric),
            current_value=current,
            projected_end_value=projected_end,
            risk_level=risk_level,
            threshold=forecast.threshold,
            threshold_crossing_date=crossing,
            days_to_threshold=days,
            trend_slope=forecast.slope,
        )

    def analyze_all(
        self,
        df: pd.DataFrame,
        forecasts: dict[str, ForecastResult],
    ) -> RiskReport:
        """Build comprehensive risk report for all metrics."""
        metrics_risk = []
        reference_date = df["date"].iloc[-1].to_pydatetime()

        for metric in SUPPORTED_METRICS:
            if metric not in forecasts:
                continue
            metrics_risk.append(
                self.analyze_single(df, forecasts[metric], reference_date)
            )

        highest = max(metrics_risk, key=lambda m: _RISK_ORDER[m.risk_level])
        crossings = [m for m in metrics_risk if m.threshold_crossing_date]
        crossings.sort(key=lambda m: m.threshold_crossing_date)

        first_metric = crossings[0] if crossings else None
        overall = highest.risk_level

        return RiskReport(
            metrics=metrics_risk,
            highest_risk_metric=highest.metric,
            first_threshold_metric=first_metric.metric if first_metric else None,
            first_threshold_date=(
                first_metric.threshold_crossing_date if first_metric else None
            ),
            overall_risk=overall,
        )

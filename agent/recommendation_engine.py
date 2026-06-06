"""Business recommendation engine for capacity planning."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from agent.risk_analyzer import MetricRisk, RiskReport
from forecasting.linear_forecast import ForecastResult
from utils.constants import METRIC_DISPLAY_NAMES, RiskLevel


@dataclass
class Recommendation:
    """Actionable capacity planning recommendation."""

    risk_level: RiskLevel
    metric: str
    display_name: str
    action: str
    timeline_days: Optional[int]
    capacity_increase_pct: Optional[float]
    priority: str
    rationale: str


class RecommendationEngine:
    """Generates business recommendations from risk and forecast data."""

    def generate(
        self,
        risk_report: RiskReport,
        forecasts: dict[str, ForecastResult],
        query_intent: str = "general",
    ) -> list[Recommendation]:
        """Generate prioritized recommendations for all at-risk metrics."""
        recommendations = []

        for metric_risk in risk_report.metrics:
            forecast = forecasts.get(metric_risk.metric)
            rec = self._build_recommendation(metric_risk, forecast)
            if rec:
                recommendations.append(rec)

        recommendations.sort(
            key=lambda r: (
                {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[r.risk_level.value],
                r.timeline_days if r.timeline_days is not None else 9999,
            )
        )
        return recommendations

    def _build_recommendation(
        self,
        risk: MetricRisk,
        forecast: Optional[ForecastResult],
    ) -> Optional[Recommendation]:
        if risk.risk_level == RiskLevel.LOW and risk.projected_end_value < risk.threshold:
            return Recommendation(
                risk_level=risk.risk_level,
                metric=risk.metric,
                display_name=risk.display_name,
                action="Continue monitoring; no immediate action required.",
                timeline_days=None,
                capacity_increase_pct=None,
                priority="P3 - Monitor",
                rationale=(
                    f"{risk.display_name} is at {risk.current_value:.1f}% with stable growth. "
                    f"Projected to reach {risk.projected_end_value:.1f}% — below threshold."
                ),
            )

        capacity_pct = self._calculate_capacity_increase(risk)
        timeline = risk.days_to_threshold
        action = self._format_action(risk, capacity_pct, timeline)
        priority = self._priority_label(risk.risk_level, timeline)

        return Recommendation(
            risk_level=risk.risk_level,
            metric=risk.metric,
            display_name=risk.display_name,
            action=action,
            timeline_days=timeline,
            capacity_increase_pct=capacity_pct,
            priority=priority,
            rationale=self._build_rationale(risk, forecast),
        )

    @staticmethod
    def _calculate_capacity_increase(risk: MetricRisk) -> float:
        headroom_needed = max(0, risk.projected_end_value - risk.threshold + 10)
        if risk.current_value <= 0:
            return 30.0
        return round(min(50.0, max(15.0, headroom_needed / risk.current_value * 100)), 1)

    @staticmethod
    def _format_action(
        risk: MetricRisk,
        capacity_pct: float,
        timeline: Optional[int],
    ) -> str:
        resource = risk.display_name.replace(" Usage", "").lower()
        if timeline is not None and timeline <= 90:
            return (
                f"Increase {resource} capacity by {capacity_pct:.0f}% "
                f"within next {timeline} days."
            )
        if risk.risk_level == RiskLevel.CRITICAL:
            return (
                f"Immediately expand {resource} capacity by {capacity_pct:.0f}% "
                f"— currently at {risk.current_value:.1f}%."
            )
        return (
            f"Plan {resource} capacity expansion of {capacity_pct:.0f}% "
            f"before utilization reaches {risk.threshold:.0f}%."
        )

    @staticmethod
    def _priority_label(risk: RiskLevel, timeline: Optional[int]) -> str:
        if risk == RiskLevel.CRITICAL:
            return "P0 - Immediate"
        if risk == RiskLevel.HIGH and timeline is not None and timeline <= 45:
            return "P1 - Urgent"
        if risk == RiskLevel.HIGH:
            return "P1 - High"
        if risk == RiskLevel.MEDIUM:
            return "P2 - Planned"
        return "P3 - Monitor"

    @staticmethod
    def _build_rationale(
        risk: MetricRisk,
        forecast: Optional[ForecastResult],
    ) -> str:
        parts = [
            f"Current {risk.display_name}: {risk.current_value:.1f}%.",
            f"Projected end-of-forecast: {risk.projected_end_value:.1f}%.",
        ]
        if risk.threshold_crossing_date:
            parts.append(
                f"Expected to cross {risk.threshold:.0f}% threshold on "
                f"{risk.threshold_crossing_date.strftime('%Y-%m-%d')}."
            )
        if forecast:
            parts.append(
                f"Trend slope: {forecast.slope:.3f}% per month "
                f"({forecast.model_type} model)."
            )
        return " ".join(parts)

    def primary_recommendation(self, recommendations: list[Recommendation]) -> Optional[Recommendation]:
        """Return the highest-priority recommendation."""
        return recommendations[0] if recommendations else None

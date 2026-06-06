"""Final answer generation using Gemini and professional structured fallbacks."""

import json
from datetime import datetime
from typing import Any, Optional

from agent.recommendation_engine import Recommendation
from agent.risk_analyzer import RiskReport
from forecasting.confidence import ConfidenceScore
from forecasting.linear_forecast import ForecastResult
from prompts.gemini_prompts import ANSWER_GENERATION_PROMPT
from utils.constants import METRIC_DISPLAY_NAMES, QueryIntent
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)


class AnswerGenerator:
    """Generates professional, human-readable answers from analysis results."""

    def __init__(self, api_key: Optional[str] = None):
        self.gemini = GeminiClient(api_key=api_key)

    def generate(
        self,
        query: str,
        forecasts: dict[str, ForecastResult],
        risk_report: RiskReport,
        confidence: ConfidenceScore,
        recommendations: list[Recommendation],
        intent: str,
    ) -> str:
        """Produce final natural language answer with professional formatting."""
        analysis = self._build_analysis_payload(
            query, forecasts, risk_report, confidence, recommendations, intent
        )

        try:
            if self.gemini.is_configured:
                prompt = ANSWER_GENERATION_PROMPT.format(
                    query=query,
                    analysis_json=json.dumps(analysis, indent=2, default=str),
                )
                return self.gemini.generate(prompt)
        except Exception as exc:
            logger.warning("Gemini answer generation failed: %s", exc)

        return self._professional_answer(query, forecasts, risk_report, confidence, recommendations, intent)

    def _build_analysis_payload(
        self,
        query: str,
        forecasts: dict[str, ForecastResult],
        risk_report: RiskReport,
        confidence: ConfidenceScore,
        recommendations: list[Recommendation],
        intent: str,
    ) -> dict[str, Any]:
        forecast_data = {}
        for metric, fc in forecasts.items():
            forecast_data[metric] = {
                "display_name": METRIC_DISPLAY_NAMES.get(metric, metric),
                "current": fc.historical_values[-1] if fc.historical_values else None,
                "forecast_end": fc.forecast_values[-1] if fc.forecast_values else None,
                "threshold": fc.threshold,
                "threshold_crossing": (
                    fc.threshold_crossing_date.isoformat()
                    if fc.threshold_crossing_date
                    else None
                ),
                "model": fc.model_type,
                "r_squared": fc.r_squared,
                "slope": fc.slope,
            }

        primary_rec = recommendations[0] if recommendations else None
        return {
            "query": query,
            "intent": intent,
            "forecasts": forecast_data,
            "risk": risk_report.to_dict(),
            "confidence": {
                "overall": confidence.overall,
                "label": confidence.label,
            },
            "primary_recommendation": (
                {
                    "risk_level": primary_rec.risk_level.value,
                    "action": primary_rec.action,
                    "priority": primary_rec.priority,
                    "rationale": primary_rec.rationale,
                }
                if primary_rec
                else None
            ),
        }

    def _professional_answer(
        self,
        query: str,
        forecasts: dict[str, ForecastResult],
        risk_report: RiskReport,
        confidence: ConfidenceScore,
        recommendations: list[Recommendation],
        intent: str,
    ) -> str:
        """Generate a professional, well-structured response with executive summary."""
        lines = []
        
        # Build executive summary header
        lines.append("📋 **Executive Summary**")
        lines.append("")
        
        fc_list = list(forecasts.values())
        
        # ========== INTENT-BASED RESPONSE ==========
        if intent == "threshold_crossing":
            lines.extend(self._threshold_crossing_response(fc_list))
        elif intent == "compare_resources":
            lines.extend(self._compare_resources_response(forecasts, risk_report))
        elif intent == "risk_analysis":
            lines.extend(self._risk_analysis_response(forecasts, risk_report))
        elif intent == "what_if":
            lines.extend(self._what_if_response(fc_list))
        elif intent == "forecast_period":
            lines.extend(self._forecast_period_response(forecasts))
        elif intent == "general":
            lines.extend(self._general_response(forecasts, risk_report))
        else:
            # Fallback: comprehensive status report
            lines.extend(self._comprehensive_report(forecasts, risk_report, recommendations))

        # ========== CURRENT STATUS OVERVIEW ==========
        lines.append("")
        lines.append("📊 **Current Status**")
        lines.append("")
        for fc in fc_list:
            name = METRIC_DISPLAY_NAMES.get(fc.metric, fc.metric)
            current_val = fc.historical_values[-1] if fc.historical_values else 0
            end_val = fc.forecast_values[-1] if fc.forecast_values else 0
            trend = "↑" if fc.slope > 0 else "↓" if fc.slope < 0 else "→"
            
            # Color-code status
            status_emoji = "✅" if current_val < 60 else "⚠️" if current_val < 80 else "🔴"
            lines.append(
                f"{status_emoji} **{name}**: {current_val:.1f}% currently → "
                f"{end_val:.1f}% projected ({trend} slope: {fc.slope:.3f}%/mo)"
            )

        # ========== THRESHOLD ANALYSIS ==========
        threshold_metrics = [fc for fc in fc_list if fc.threshold_crossing_date]
        if threshold_metrics:
            lines.append("")
            lines.append("⏰ **Threshold Crossings**")
            lines.append("")
            for fc in sorted(threshold_metrics, key=lambda x: x.threshold_crossing_date):
                name = METRIC_DISPLAY_NAMES.get(fc.metric, fc.metric)
                crossing_dt = fc.threshold_crossing_date
                now = datetime.now()
                # Ensure both operands are datetime.datetime for subtraction
                if isinstance(crossing_dt, datetime) and isinstance(now, datetime):
                    days_until = (crossing_dt - now).days
                else:
                    days_until = 0
                lines.append(
                    f"• **{name}** will cross {fc.threshold:.0f}% on "
                    f"**{crossing_dt.strftime('%b %d, %Y')}** "
                    f"({days_until} days from now)"
                )
        else:
            crossing_any = any(
                fc.threshold_crossing_date for fc in fc_list
            )
            if not crossing_any:
                lines.append("")
                lines.append("✅ **Threshold Analysis**")
                lines.append("")
                lines.append("No resources are projected to exceed their thresholds within the forecast horizon.")

        # ========== CONFIDENCE & RISK ==========
        lines.append("")
        lines.append("🎯 **Forecast Quality**")
        lines.append("")
        lines.append(f"• **Confidence Score**: {confidence.label} ({confidence.overall * 100:.0f}%)")
        lines.append(f"• **Overall Risk Level**: {risk_report.overall_risk.value}")
        
        # Model performance
        for fc in fc_list:
            name = METRIC_DISPLAY_NAMES.get(fc.metric, fc.metric)
            lines.append(f"• **{name} Model ({fc.model_type.upper()})**: R² = {fc.r_squared:.3f}")
        
        # ========== RECOMMENDATIONS ==========
        if recommendations:
            lines.append("")
            lines.append("💡 **Recommendations**")
            lines.append("")
            
            # Primary recommendation with priority badge
            primary = recommendations[0]
            priority_badges = {
                "P0": "🔴 CRITICAL",
                "P1": "🟠 HIGH",
                "P2": "🟡 MEDIUM",
                "P3": "🟢 LOW",
            }
            badge = "🔴 CRITICAL"
            for key, value in priority_badges.items():
                if key in primary.priority:
                    badge = value
                    break
            
            lines.append(f"**Priority {badge}**")
            lines.append(f"> {primary.action}")
            lines.append(f"> *Rationale*: {primary.rationale}")
            
            # Secondary recommendations
            if len(recommendations) > 1:
                lines.append("")
                lines.append("**Additional Recommendations:**")
                for i, rec in enumerate(recommendations[1:], 1):
                    lines.append(f"  {i}. **{rec.display_name}** ({rec.risk_level.value}): {rec.action}")

        # ========== TIME-BASED CONTEXT ==========
        lines.append("")
        lines.append("---")
        lines.append(f"*Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                     f"Forecast horizon: {self._get_forecast_horizon(fc_list)} months")

        return "\n".join(lines)

    def _threshold_crossing_response(self, fc_list: list[ForecastResult]) -> list[str]:
        """Response for 'when will X hit Y%' queries."""
        lines = []
        now = datetime.now()
        for fc in fc_list:
            name = METRIC_DISPLAY_NAMES.get(fc.metric, fc.metric)
            current_val = fc.historical_values[-1] if fc.historical_values else 0
            
            if fc.threshold_crossing_date:
                crossing_dt = fc.threshold_crossing_date
                # Ensure both operands are datetime.datetime for subtraction
                if isinstance(crossing_dt, datetime) and isinstance(now, datetime):
                    days_remaining = (crossing_dt - now).days
                else:
                    days_remaining = 0
                lines.append(
                    f"🔮 **{name}** is currently at **{current_val:.1f}%** and is projected "
                    f"to reach **{fc.threshold:.0f}%** threshold on "
                    f"**{crossing_dt.strftime('%B %d, %Y')}**."
                )
                lines.append(f"")
                lines.append(f"⏱️ That's **{days_remaining} days** from now at the current growth rate "
                            f"of {fc.slope:.2f}% per month.")
                
                # Urgency indicator
                if days_remaining <= 30:
                    lines.append(f"🔴 **URGENT**: Threshold breach expected within 30 days!")
                elif days_remaining <= 90:
                    lines.append(f"🟡 **PLANNED**: Threshold breach expected within 90 days. Start planning mitigation.")
                else:
                    lines.append(f"🟢 **MONITOR**: Threshold breach is beyond 90 days. Regular monitoring recommended.")
            else:
                end_val = fc.forecast_values[-1] if fc.forecast_values else current_val
                lines.append(
                    f"✅ **{name}** is currently at **{current_val:.1f}%** and is **not** projected "
                    f"to reach {fc.threshold:.0f}% within the forecast horizon. "
                    f"Projected end value: {end_val:.1f}%."
                )
        return lines

    def _compare_resources_response(
        self, forecasts: dict[str, ForecastResult], risk_report: RiskReport
    ) -> list[str]:
        """Response for 'which resource exceeds first' queries."""
        lines = []
        now = datetime.now()
        if risk_report.first_threshold_metric and risk_report.first_threshold_date:
            name = METRIC_DISPLAY_NAMES.get(
                risk_report.first_threshold_metric,
                risk_report.first_threshold_metric,
            )
            first_date = risk_report.first_threshold_date
            # Ensure both operands are datetime.datetime for subtraction
            if isinstance(first_date, datetime) and isinstance(now, datetime):
                days_remaining = (first_date - now).days
            else:
                days_remaining = 0
            lines.append(
                f"🏆 **{name}** will be the first resource to exceed its threshold, "
                f"projected on **{first_date.strftime('%B %d, %Y')}** "
                f"({days_remaining} days from now)."
            )
            lines.append("")
            lines.append("**📅 Resource Threshold Crossing Timeline:**")
            lines.append("")
            
            # Build timeline for all resources
            crossings = []
            for metric, fc in forecasts.items():
                name = METRIC_DISPLAY_NAMES.get(metric, metric)
                if fc.threshold_crossing_date:
                    crossings.append((fc.threshold_crossing_date, name, fc.metric))
            
            crossings.sort(key=lambda x: x[0])
            for i, (date, name, metric) in enumerate(crossings, 1):
                current = forecasts[metric].historical_values[-1] if forecasts[metric].historical_values else 0
                lines.append(
                    f"  {i}. **{name}** — **{date.strftime('%b %d, %Y')}** "
                    f"(currently {current:.1f}%)"
                )
        else:
            lines.append("✅ No resource is projected to exceed its threshold within the forecast horizon.")
        return lines

    def _risk_analysis_response(
        self, forecasts: dict[str, ForecastResult], risk_report: RiskReport
    ) -> list[str]:
        """Response for risk analysis queries."""
        lines = []
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
        emoji = risk_emoji.get(risk_report.overall_risk.value, "⚪")
        lines.append(
            f"{emoji} **Overall Risk Level**: {risk_report.overall_risk.value}"
        )
        lines.append("")
        lines.append("**📊 Per-Resource Risk Assessment:**")
        lines.append("")
        for m in risk_report.metrics:
            risk_em = risk_emoji.get(m.risk_level.value, "⚪")
            days_str = ""
            if m.days_to_threshold is not None:
                days_str = f" | Threshold in {m.days_to_threshold} days"
            lines.append(
                f"  {risk_em} **{m.display_name}**: {m.current_value:.1f}% → "
                f"{m.projected_end_value:.1f}% — **{m.risk_level.value}** risk{days_str}"
            )
        return lines

    def _what_if_response(self, fc_list: list[ForecastResult]) -> list[str]:
        """Response for scenario analysis / what-if queries."""
        lines = []
        now = datetime.now()
        for fc in fc_list:
            name = METRIC_DISPLAY_NAMES.get(fc.metric, fc.metric)
            current_val = fc.historical_values[-1] if fc.historical_values else 0
            end_val = fc.forecast_values[-1] if fc.forecast_values else 0
            
            if fc.threshold_crossing_date:
                crossing_dt = fc.threshold_crossing_date
                if isinstance(crossing_dt, datetime) and isinstance(now, datetime):
                    days_remaining = (crossing_dt - now).days
                else:
                    days_remaining = 0
                lines.append(
                    f"🔮 **Scenario Analysis — {name}**"
                )
                lines.append("")
                lines.append(
                    f"Under the adjusted growth scenario (slope: {fc.slope:.3f}%/mo), "
                    f"**{name}** reaches **{fc.threshold:.0f}%** by "
                    f"**{crossing_dt.strftime('%B %d, %Y')}** "
                    f"({days_remaining} days)."
                )
                lines.append("")
                lines.append(f"  • Current: {current_val:.1f}%")
                lines.append(f"  • Projected (end): {end_val:.1f}%")
                lines.append(f"  • Growth rate: {fc.slope:.3f}%/month")
                lines.append(f"  • Model R²: {fc.r_squared:.3f}")
            else:
                lines.append(
                    f"📈 **{name}** under the adjusted scenario: "
                    f"current {current_val:.1f}% → projected {end_val:.1f}%. "
                    f"No threshold breach expected."
                )
        return lines

    def _forecast_period_response(self, forecasts: dict[str, ForecastResult]) -> list[str]:
        """Response for standard forecast queries."""
        lines = []
        lines.append("**📈 Forecast Projections**")
        lines.append("")
        
        for metric, fc in forecasts.items():
            name = METRIC_DISPLAY_NAMES.get(metric, metric)
            current_val = fc.historical_values[-1] if fc.historical_values else 0
            end_val = fc.forecast_values[-1] if fc.forecast_values else 0
            
            # Determine direction emoji
            direction = "📈" if end_val > current_val else "📉" if end_val < current_val else "➡️"
            
            lines.append(
                f"  {direction} **{name}**: {current_val:.1f}% → **{end_val:.1f}%** "
                f"(rate: {fc.slope:+.3f}%/mo)"
            )
            
            # Add monthly breakdown for first metric
            if metric == list(forecasts.keys())[0] and fc.forecast_values:
                lines.append("")
                lines.append(f"    *Monthly projection for {name} (top 5):*")
                for i, val in enumerate(fc.forecast_values[:5], 1):
                    lines.append(f"      Month {i}: {val:.1f}%")
                if len(fc.forecast_values) > 5:
                    lines.append(f"      ... and {len(fc.forecast_values) - 5} more months")
        
        return lines

    def _general_response(self, forecasts: dict[str, ForecastResult], risk_report: RiskReport) -> list[str]:
        """Response for general queries."""
        lines = []
        lines.append("**📊 System Status Overview**")
        lines.append("")
        for metric, fc in forecasts.items():
            name = METRIC_DISPLAY_NAMES.get(metric, metric)
            current_val = fc.historical_values[-1] if fc.historical_values else 0
            end_val = fc.forecast_values[-1] if fc.forecast_values else 0
            lines.append(
                f"  • **{name}**: {current_val:.1f}% now, trending to {end_val:.1f}% "
                f"(R² = {fc.r_squared:.3f})"
            )
        
        # Find the most critical metric
        most_critical = min(
            risk_report.metrics,
            key=lambda m: {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(
                m.risk_level.value, 0
            ),
            default=None,
        )
        if most_critical and most_critical.risk_level.value in ("HIGH", "CRITICAL"):
            lines.append("")
            lines.append(f"⚠️ **Attention Needed**: {most_critical.display_name} has "
                        f"{most_critical.risk_level.value} risk level.")
        
        return lines

    def _comprehensive_report(
        self,
        forecasts: dict[str, ForecastResult],
        risk_report: RiskReport,
        recommendations: list[Recommendation],
    ) -> list[str]:
        """Comprehensive system health report."""
        lines = self._risk_analysis_response(forecasts, risk_report)
        
        if recommendations:
            lines.append("")
            lines.append("**🎯 Recommended Actions (Priority Order):**")
            for i, rec in enumerate(recommendations[:5], 1):
                lines.append(f"  {i}. [{rec.priority}] {rec.display_name}: {rec.action}")
        
        return lines

    @staticmethod
    def _get_forecast_horizon(fc_list: list[ForecastResult]) -> int:
        """Get the forecast horizon in months."""
        if fc_list and fc_list[0].forecast_values:
            return len(fc_list[0].forecast_values)
        return 0
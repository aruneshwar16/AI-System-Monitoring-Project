"""AI Agent controller orchestrating the capacity forecasting workflow."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

from agent.answer_generator import AnswerGenerator
from agent.nl_parser import NLParser, ParsedQuery
from agent.recommendation_engine import Recommendation, RecommendationEngine
from agent.risk_analyzer import RiskAnalyzer, RiskReport
from forecasting.arima_forecast import forecast_metric_arima
from forecasting.confidence import ConfidenceScore, calculate_confidence
from forecasting.linear_forecast import (
    ForecastResult,
    compare_metrics_threshold_crossing,
    forecast_metric_linear,
)
from prompts.gemini_prompts import VALIDATION_PROMPT
from utils.constants import ForecastModel, QueryIntent, SUPPORTED_METRICS
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Complete response from the agent pipeline."""

    query: str
    parsed: ParsedQuery
    forecasts: dict[str, ForecastResult] = field(default_factory=dict)
    risk_report: Optional[RiskReport] = None
    confidence: Optional[ConfidenceScore] = None
    recommendations: list[Recommendation] = field(default_factory=list)
    answer: str = ""
    validation_passed: bool = True
    validation_issues: list[str] = field(default_factory=list)

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.parsed.intent.value,
            "answer": self.answer,
            "confidence": self.confidence.overall if self.confidence else None,
            "risk_level": (
                self.risk_report.overall_risk.value if self.risk_report else None
            ),
            "recommendation": (
                self.recommendations[0].action if self.recommendations else None
            ),
        }


class AgentController:
    """
    AI Agent loop:
    Understand Query -> Select Metric -> Forecast -> Validate -> Generate Answer
    """

    def __init__(
        self,
        model: ForecastModel = ForecastModel.LINEAR,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.parser = NLParser(api_key=api_key)
        self.risk_analyzer = RiskAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        self.answer_generator = AnswerGenerator(api_key=api_key)
        self.gemini = GeminiClient(api_key=api_key)

    def run(self, query: str, df: pd.DataFrame) -> AgentResponse:
        """Execute the full agent workflow."""
        logger.info("Agent started for query: %s", query)

        # Step 1: Understand Query (Intent Detection + Threshold Extraction)
        parsed = self.parser.parse(query)
        logger.info(
            "Parsed intent=%s metrics=%s threshold=%.1f months=%d",
            parsed.intent.value,
            parsed.metrics,
            parsed.threshold,
            parsed.forecast_months,
        )

        # Step 2: Select Metric & Step 3: Forecast Engine
        forecasts = self._run_forecasts(df, parsed)

        # Step 4: Risk Analysis
        risk_report = self.risk_analyzer.analyze_all(df, forecasts)

        # Step 5: Confidence Score
        primary_metric = parsed.metrics[0]
        primary_forecast = forecasts[primary_metric]
        confidence = calculate_confidence(
            forecast=primary_forecast,
            data_points=len(df),
            forecast_months=parsed.forecast_months,
            parse_confidence=parsed.parse_confidence,
        )

        # Step 6: Recommendation Engine
        recommendations = self.recommendation_engine.generate(
            risk_report, forecasts, parsed.intent.value
        )

        # Build interim analysis for validation
        interim = self._build_interim_analysis(
            parsed, forecasts, risk_report, confidence, recommendations
        )

        # Step 7: Validate
        validation_passed, issues, adjusted = self._validate(interim)
        if adjusted != 1.0:
            confidence = calculate_confidence(
                forecast=primary_forecast,
                data_points=len(df),
                forecast_months=parsed.forecast_months,
                parse_confidence=parsed.parse_confidence,
                validation_adjustment=adjusted,
            )

        # Step 8: Generate Answer
        answer = self.answer_generator.generate(
            query=query,
            forecasts=forecasts,
            risk_report=risk_report,
            confidence=confidence,
            recommendations=recommendations,
            intent=parsed.intent.value,
        )

        response = AgentResponse(
            query=query,
            parsed=parsed,
            forecasts=forecasts,
            risk_report=risk_report,
            confidence=confidence,
            recommendations=recommendations,
            answer=answer,
            validation_passed=validation_passed,
            validation_issues=issues,
        )

        logger.info("Agent completed. Risk=%s Confidence=%.2f", risk_report.overall_risk.value, confidence.overall)
        return response

    def _run_forecasts(self, df: pd.DataFrame, parsed: ParsedQuery) -> dict[str, ForecastResult]:
        """Execute forecasting for selected metrics."""
        forecasts: dict[str, ForecastResult] = {}
        forecast_fn = (
            forecast_metric_arima
            if self.model == ForecastModel.ARIMA
            else forecast_metric_linear
        )

        metrics_to_forecast = parsed.metrics
        if parsed.intent == QueryIntent.COMPARE_RESOURCES:
            metrics_to_forecast = list(SUPPORTED_METRICS)

        for metric in metrics_to_forecast:
            multiplier = parsed.growth_multiplier if parsed.intent == QueryIntent.WHAT_IF else 1.0
            if parsed.intent == QueryIntent.WHAT_IF and metric != parsed.metrics[0]:
                multiplier = 1.0

            forecasts[metric] = forecast_fn(
                df=df,
                metric=metric,
                forecast_months=parsed.forecast_months,
                threshold=parsed.threshold,
                growth_multiplier=multiplier,
            )

        if parsed.intent in (QueryIntent.RISK_ANALYSIS, QueryIntent.COMPARE_RESOURCES):
            for metric in SUPPORTED_METRICS:
                if metric not in forecasts:
                    forecasts[metric] = forecast_fn(
                        df=df,
                        metric=metric,
                        forecast_months=parsed.forecast_months,
                        threshold=parsed.threshold,
                    )

        return forecasts

    def _build_interim_analysis(
        self,
        parsed: ParsedQuery,
        forecasts: dict[str, ForecastResult],
        risk_report: RiskReport,
        confidence: ConfidenceScore,
        recommendations: list[Recommendation],
    ) -> dict[str, Any]:
        return {
            "intent": parsed.intent.value,
            "threshold": parsed.threshold,
            "forecast_months": parsed.forecast_months,
            "risk": risk_report.to_dict(),
            "confidence": confidence.overall,
            "forecasts": {
                m: {
                    "end_value": fc.forecast_values[-1] if fc.forecast_values else None,
                    "threshold_crossing": (
                        fc.threshold_crossing_date.isoformat()
                        if fc.threshold_crossing_date
                        else None
                    ),
                }
                for m, fc in forecasts.items()
            },
            "recommendations": [r.action for r in recommendations[:3]],
        }

    def _validate(self, analysis: dict[str, Any]) -> tuple[bool, list[str], float]:
        """Validate analysis consistency using Gemini or rule-based checks."""
        issues = []
        adjusted = 1.0

        risk_data = analysis.get("risk", {})
        for m in risk_data.get("metrics", []):
            if m["current_value"] < 0 or m["current_value"] > 100:
                issues.append(f"Invalid current value for {m['metric']}")
            if m["projected_end_value"] < 0 or m["projected_end_value"] > 100:
                issues.append(f"Invalid projected value for {m['metric']}")

        if analysis.get("confidence", 0) < 0 or analysis.get("confidence", 0) > 1:
            issues.append("Confidence out of range")

        try:
            if self.gemini.is_configured:
                prompt = VALIDATION_PROMPT.format(
                    analysis_json=json.dumps(analysis, indent=2, default=str)
                )
                text = self.gemini.generate(prompt)
                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()
                result = json.loads(text)
                if not result.get("is_valid", True):
                    issues.extend(result.get("issues", []))
                adjusted = float(result.get("adjusted_confidence", 1.0))
        except Exception as exc:
            logger.debug("Validation via Gemini skipped: %s", exc)

        passed = len(issues) == 0
        if issues:
            adjusted = min(adjusted, 0.85)
        return passed, issues, adjusted

    def get_comparison_summary(
        self, df: pd.DataFrame, threshold: float = 80.0, forecast_months: int = 12
    ) -> list[tuple[str, Any, float]]:
        """Utility for resource comparison queries."""
        return compare_metrics_threshold_crossing(df, forecast_months, threshold)

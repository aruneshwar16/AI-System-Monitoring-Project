"""Unit tests for natural language query parser."""

import pytest

from agent.nl_parser import NLParser, ParsedQuery
from utils.constants import (
    METRIC_CPU,
    METRIC_DISK,
    METRIC_MEMORY,
    QueryIntent,
)


class TestRuleBasedParser:
    """Tests for rule-based fallback parsing (no API required)."""

    @pytest.fixture
    def parser(self):
        return NLParser(api_key="")

    def test_threshold_crossing_disk(self, parser):
        result = parser._parse_rule_based("When will disk usage hit 80%?")
        assert result.intent == QueryIntent.THRESHOLD_CROSSING
        assert METRIC_DISK in result.metrics
        assert result.threshold == 80.0

    def test_forecast_cpu_six_months(self, parser):
        result = parser._parse_rule_based("Forecast CPU usage for next 6 months.")
        assert result.intent == QueryIntent.FORECAST_PERIOD
        assert METRIC_CPU in result.metrics
        assert result.forecast_months == 6

    def test_compare_resources(self, parser):
        result = parser._parse_rule_based("Which resource will exceed threshold first?")
        assert result.intent == QueryIntent.COMPARE_RESOURCES
        assert len(result.metrics) == 3

    def test_risk_analysis(self, parser):
        result = parser._parse_rule_based("Show risk analysis for all resources.")
        assert result.intent == QueryIntent.RISK_ANALYSIS
        assert METRIC_CPU in result.metrics
        assert METRIC_MEMORY in result.metrics
        assert METRIC_DISK in result.metrics

    def test_what_if_growth(self, parser):
        result = parser._parse_rule_based("What if CPU grows 20% faster?")
        assert result.intent == QueryIntent.WHAT_IF
        assert METRIC_CPU in result.metrics
        assert result.growth_multiplier == pytest.approx(1.2, rel=0.01)

    def test_extract_threshold_custom(self, parser):
        result = parser._parse_rule_based("When will memory hit 90%?")
        assert result.threshold == 90.0
        assert METRIC_MEMORY in result.metrics

    def test_extract_months_twelve(self, parser):
        result = parser._parse_rule_based("Forecast disk for next 12 months")
        assert result.forecast_months == 12
        assert METRIC_DISK in result.metrics

    def test_parse_empty_query_raises(self, parser):
        with pytest.raises(ValueError, match="empty"):
            parser.parse("   ")

    def test_parse_returns_parsed_query(self, parser):
        result = parser.parse("Forecast memory for 3 months")
        assert isinstance(result, ParsedQuery)
        assert result.source == "rule_based"
        assert result.forecast_months == 3

    def test_default_threshold(self, parser):
        result = parser._parse_rule_based("When will CPU exceed threshold?")
        assert result.threshold == 80.0

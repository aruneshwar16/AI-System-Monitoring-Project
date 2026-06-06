"""Unit tests for forecasting engines."""

import pandas as pd
import pytest

from forecasting.arima_forecast import forecast_metric_arima
from forecasting.confidence import calculate_confidence, risk_from_utilization
from forecasting.linear_forecast import (
    compare_metrics_threshold_crossing,
    forecast_metric_linear,
)
from utils.constants import METRIC_CPU, METRIC_DISK, RiskLevel


class TestLinearForecast:
    def test_forecast_returns_expected_fields(self, sample_df):
        result = forecast_metric_linear(sample_df, METRIC_CPU, forecast_months=6)
        assert result.metric == METRIC_CPU
        assert len(result.historical_values) == len(sample_df)
        assert len(result.forecast_values) == 6
        assert len(result.forecast_dates) == 6
        assert result.model_type == "linear"

    def test_forecast_values_in_valid_range(self, sample_df):
        result = forecast_metric_linear(sample_df, METRIC_DISK, forecast_months=6)
        for val in result.forecast_values:
            assert 0 <= val <= 100

    def test_r_squared_calculated(self, sample_df):
        result = forecast_metric_linear(sample_df, METRIC_CPU)
        assert 0 <= result.r_squared <= 1

    def test_growth_multiplier_increases_slope(self, sample_df):
        baseline = forecast_metric_linear(sample_df, METRIC_CPU, growth_multiplier=1.0)
        accelerated = forecast_metric_linear(sample_df, METRIC_CPU, growth_multiplier=1.2)
        assert accelerated.slope > baseline.slope
        assert accelerated.forecast_values[-1] >= baseline.forecast_values[-1]

    def test_threshold_crossing_detected_for_growing_disk(self, sample_df):
        result = forecast_metric_linear(
            sample_df, METRIC_DISK, forecast_months=12, threshold=80.0
        )
        assert result.threshold_crossing_date is not None

    def test_compare_metrics_ordering(self, sample_df):
        comparison = compare_metrics_threshold_crossing(
            sample_df, forecast_months=12, threshold=80.0
        )
        assert len(comparison) == 3
        dated = [c for c in comparison if c[1] is not None]
        if len(dated) >= 2:
            assert dated[0][1] <= dated[1][1]

    def test_invalid_metric_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported"):
            forecast_metric_linear(sample_df, "network_usage")


class TestArimaForecast:
    def test_arima_forecast_produces_results(self, sample_df):
        result = forecast_metric_arima(sample_df, METRIC_CPU, forecast_months=6)
        assert result.model_type == "arima"
        assert len(result.forecast_values) == 6

    def test_arima_values_bounded(self, minimal_df):
        result = forecast_metric_arima(minimal_df, METRIC_DISK, forecast_months=3)
        for val in result.forecast_values:
            assert 0 <= val <= 100


class TestConfidence:
    def test_confidence_score_range(self, sample_df):
        forecast = forecast_metric_linear(sample_df, METRIC_CPU)
        score = calculate_confidence(forecast, data_points=len(sample_df), forecast_months=6)
        assert 0 <= score.overall <= 1
        assert score.label in ("Very High", "High", "Moderate", "Low", "Very Low")

    def test_longer_horizon_lowers_confidence(self, sample_df):
        forecast = forecast_metric_linear(sample_df, METRIC_CPU)
        short = calculate_confidence(forecast, len(sample_df), forecast_months=3)
        long = calculate_confidence(forecast, len(sample_df), forecast_months=18)
        assert short.forecast_horizon >= long.forecast_horizon


class TestRiskLevels:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (50, RiskLevel.LOW),
            (65, RiskLevel.MEDIUM),
            (85, RiskLevel.HIGH),
            (95, RiskLevel.CRITICAL),
        ],
    )
    def test_risk_from_utilization(self, value, expected):
        assert risk_from_utilization(value) == expected

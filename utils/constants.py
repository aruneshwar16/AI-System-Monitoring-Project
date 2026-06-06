"""Application-wide constants for the Capacity Forecaster."""

from enum import Enum
from typing import Final

# Supported metrics (column names in CSV)
METRIC_CPU: Final[str] = "cpu_usage"
METRIC_MEMORY: Final[str] = "memory_usage"
METRIC_DISK: Final[str] = "disk_usage"

SUPPORTED_METRICS: Final[tuple[str, ...]] = (
    METRIC_CPU,
    METRIC_MEMORY,
    METRIC_DISK,
)

METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    METRIC_CPU: "CPU Usage",
    METRIC_MEMORY: "Memory Usage",
    METRIC_DISK: "Disk Usage",
}

# Required CSV columns
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("date",) + SUPPORTED_METRICS

# Default thresholds (percentage)
DEFAULT_THRESHOLDS: Final[dict[str, float]] = {
    METRIC_CPU: 80.0,
    METRIC_MEMORY: 80.0,
    METRIC_DISK: 80.0,
}

# Risk level boundaries (current utilization %)
RISK_LOW_MAX: Final[float] = 60.0
RISK_MEDIUM_MAX: Final[float] = 80.0
RISK_HIGH_MAX: Final[float] = 90.0


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ForecastModel(str, Enum):
    LINEAR = "linear"
    ARIMA = "arima"


class QueryIntent(str, Enum):
    THRESHOLD_CROSSING = "threshold_crossing"
    FORECAST_PERIOD = "forecast_period"
    COMPARE_RESOURCES = "compare_resources"
    RISK_ANALYSIS = "risk_analysis"
    WHAT_IF = "what_if"
    GENERAL = "general"


DEFAULT_FORECAST_MONTHS: Final[int] = 6
MAX_FORECAST_MONTHS: Final[int] = 24
DEFAULT_THRESHOLD: Final[float] = 80.0

GEMINI_MODEL: Final[str] = "gemini-2.0-flash"

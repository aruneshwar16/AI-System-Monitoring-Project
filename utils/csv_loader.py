"""CSV loading and validation utilities."""

from pathlib import Path
from typing import BinaryIO, Optional, Union

import pandas as pd

from utils.constants import REQUIRED_COLUMNS, SUPPORTED_METRICS
from utils.logger import get_logger

logger = get_logger(__name__)


class CSVLoadError(Exception):
    """Raised when CSV loading or validation fails."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("%", "pct")
        for col in df.columns
    ]
    column_map = {
        "cpu": "cpu_usage",
        "cpu_pct": "cpu_usage",
        "cpu_usage_pct": "cpu_usage",
        "memory": "memory_usage",
        "memory_pct": "memory_usage",
        "memory_usage_pct": "memory_usage",
        "disk": "disk_usage",
        "disk_pct": "disk_usage",
        "disk_usage_pct": "disk_usage",
        "timestamp": "date",
        "datetime": "date",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare metrics DataFrame."""
    df = _normalize_columns(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise CSVLoadError(
            f"Missing required columns: {missing}. "
            f"Expected: {list(REQUIRED_COLUMNS)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise CSVLoadError("Invalid date values found in 'date' column.")

    for metric in SUPPORTED_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        if df[metric].isna().any():
            raise CSVLoadError(f"Invalid numeric values in '{metric}' column.")
        if (df[metric] < 0).any() or (df[metric] > 100).any():
            raise CSVLoadError(
                f"'{metric}' values must be between 0 and 100 (percentage)."
            )

    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 3:
        raise CSVLoadError("At least 3 data points are required for forecasting.")

    logger.info("Loaded %d rows spanning %s to %s", len(df), df["date"].min(), df["date"].max())
    return df[["date"] + list(SUPPORTED_METRICS)]


def load_csv(
    source: Union[str, Path, BinaryIO],
) -> pd.DataFrame:
    """Load and validate metrics from a CSV file or file-like object."""
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise CSVLoadError(f"Failed to read CSV: {exc}") from exc
    return validate_dataframe(df)


def load_default_sample() -> pd.DataFrame:
    """Load the bundled sample metrics CSV."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_metrics.csv"
    if not sample_path.exists():
        raise CSVLoadError(f"Sample data not found at {sample_path}")
    return load_csv(sample_path)

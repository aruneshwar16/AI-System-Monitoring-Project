"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.csv_loader import load_default_sample


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return validated sample metrics DataFrame."""
    return load_default_sample()


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal dataset for unit tests."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "cpu_usage": [40.0, 42.0, 44.0, 46.0, 48.0, 50.0],
            "memory_usage": [50.0, 52.0, 54.0, 56.0, 58.0, 60.0],
            "disk_usage": [60.0, 63.0, 66.0, 69.0, 72.0, 75.0],
        }
    )

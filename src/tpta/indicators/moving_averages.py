"""Moving-average indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _window(window: int) -> int:
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    return window


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average using only current and past observations."""
    return series.astype(float).rolling(_window(window), min_periods=window).mean().rename(f"sma_{window}")


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average with an explicit warm-up period."""
    return series.astype(float).ewm(span=_window(window), adjust=False, min_periods=window).mean().rename(f"ema_{window}")


def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted moving average, with more weight on recent values."""
    window = _window(window)
    weights = np.arange(1, window + 1, dtype=float)
    return series.astype(float).rolling(window, min_periods=window).apply(lambda values: np.dot(values, weights) / weights.sum(), raw=True).rename(f"wma_{window}")


def hma(series: pd.Series, window: int) -> pd.Series:
    """Hull moving average."""
    window = _window(window)
    half = max(window // 2, 1)
    root = max(int(np.sqrt(window)), 1)
    return wma(2 * wma(series, half) - wma(series, window), root).rename(f"hma_{window}")


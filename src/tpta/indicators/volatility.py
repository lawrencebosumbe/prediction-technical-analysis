"""Volatility and range indicators."""

from __future__ import annotations

import pandas as pd


def true_range(frame: pd.DataFrame) -> pd.Series:
    """True range from high, low, and previous close."""
    previous = frame["close"].shift(1)
    values = pd.concat([frame["high"] - frame["low"], (frame["high"] - previous).abs(), (frame["low"] - previous).abs()], axis=1)
    return values.max(axis=1).rename("true_range")


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range using Wilder smoothing."""
    if window < 1:
        raise ValueError("window must be positive")
    return true_range(frame).ewm(alpha=1 / window, adjust=False, min_periods=window).mean().rename(f"atr_{window}")


def bollinger_bands(series: pd.Series, window: int = 20, deviations: float = 2.0) -> pd.DataFrame:
    """Rolling mean and upper/lower bands based on population standard deviation."""
    if window < 1 or deviations <= 0:
        raise ValueError("window and deviations must be positive")
    mean = series.astype(float).rolling(window, min_periods=window).mean()
    std = series.astype(float).rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"middle": mean, "upper": mean + deviations * std, "lower": mean - deviations * std}, index=series.index)


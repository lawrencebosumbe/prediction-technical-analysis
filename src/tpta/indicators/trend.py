"""Trend strength and channel indicators."""

from __future__ import annotations

import pandas as pd

from tpta.indicators.volatility import true_range


def adx(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index using Wilder-style exponential smoothing."""
    if window < 1:
        raise ValueError("window must be positive")
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    smooth = lambda series: series.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    tr = smooth(true_range(frame)).replace(0, float("nan"))
    plus_di, minus_di = 100 * smooth(plus) / tr, 100 * smooth(minus) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return smooth(dx).rename(f"adx_{window}")


def donchian_channels(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling high and low breakout channels."""
    if window < 1:
        raise ValueError("window must be positive")
    return pd.DataFrame({"upper": frame["high"].rolling(window, min_periods=window).max(), "lower": frame["low"].rolling(window, min_periods=window).min()}, index=frame.index)


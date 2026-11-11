"""Confirmed swing-point divergence detection."""

from __future__ import annotations

import pandas as pd


def confirmed_pivots(series: pd.Series, left: int = 3, right: int = 3, kind: str = "low") -> pd.Series:
    """Mark pivots only after `right` bars have elapsed, avoiding look-ahead."""
    if left < 1 or right < 1 or kind not in {"low", "high"}:
        raise ValueError("left/right must be positive and kind must be low or high")
    window = left + right + 1
    rolling = series.rolling(window, center=False, min_periods=window)
    # Shift by `right` to place the confirmed pivot at the time it becomes observable.
    candidates = series.shift(right)
    extrema = rolling.min().shift(0) if kind == "low" else rolling.max().shift(0)
    return candidates.where(candidates.eq(extrema))


def divergence(price: pd.Series, oscillator: pd.Series, window: int = 5) -> pd.Series:
    """Return +1 bullish and -1 bearish divergence events from local extrema."""
    if window < 1:
        raise ValueError("window must be positive")
    price, oscillator = price.align(oscillator, join="inner")
    p_low = price.shift(1).rolling(window, min_periods=window).min()
    o_low = oscillator.shift(1).rolling(window, min_periods=window).min()
    p_high = price.shift(1).rolling(window, min_periods=window).max()
    o_high = oscillator.shift(1).rolling(window, min_periods=window).max()
    bullish = (price < p_low) & (oscillator > o_low)
    bearish = (price > p_high) & (oscillator < o_high)
    return (bullish.astype(int) - bearish.astype(int)).rename("divergence")


"""Price momentum indicators."""

from __future__ import annotations

import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI with values bounded from zero to one hundred."""
    if window < 1:
        raise ValueError("window must be positive")
    change = series.astype(float).diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = -change.clip(upper=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative = gain / loss.replace(0, float("nan"))
    result = 100 - 100 / (1 + relative)
    result = result.mask((loss == 0) & (gain > 0), 100.0).mask((loss == 0) & (gain == 0), 50.0)
    return result.rename(f"rsi_{window}")


def stochastic(frame: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.DataFrame:
    """Fast %K and smoothed %D stochastic oscillator."""
    if window < 1 or smooth < 1:
        raise ValueError("windows must be positive")
    low = frame["low"].rolling(window, min_periods=window).min()
    high = frame["high"].rolling(window, min_periods=window).max()
    spread = (high - low).replace(0, float("nan"))
    k = 100 * (frame["close"] - low) / spread
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(smooth, min_periods=smooth).mean()}, index=frame.index)


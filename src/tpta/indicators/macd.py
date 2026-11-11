"""Moving Average Convergence Divergence."""

from __future__ import annotations

import pandas as pd

from tpta.indicators.moving_averages import ema


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Return MACD, signal, and histogram columns."""
    if min(fast, slow, signal) < 1 or fast >= slow:
        raise ValueError("periods must be positive and fast must be less than slow")
    line = ema(series, fast) - ema(series, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": signal_line, "histogram": line - signal_line}, index=series.index)


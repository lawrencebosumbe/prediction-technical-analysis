"""Simple trading-calendar helpers."""

from __future__ import annotations

import pandas as pd


def future_sessions(last_timestamp: pd.Timestamp, periods: int, *, crypto: bool = False) -> pd.DatetimeIndex:
    """Return future daily timestamps for a market that trades every day or weekdays."""
    if periods < 1:
        raise ValueError("periods must be positive")
    start = pd.Timestamp(last_timestamp) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start.normalize(), periods=periods if crypto else periods * 2, freq="D" if crypto else "B", tz="UTC")
    return dates[:periods]


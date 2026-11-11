"""Moving-average and MACD directional states."""

from __future__ import annotations

import pandas as pd

from tpta.signals.base import normalize_signal


def sma_crossover(frame: pd.DataFrame, fast: str = "sma_fast", slow: str = "sma_slow") -> pd.Series:
    """Return the current long/short state of two moving averages."""
    if fast not in frame or slow not in frame:
        raise KeyError(f"Expected {fast!r} and {slow!r} columns")
    values = (frame[fast] > frame[slow]).astype("int8") - (frame[fast] < frame[slow]).astype("int8")
    return normalize_signal(values.where(frame[fast].notna() & frame[slow].notna(), 0))


def macd_crossover(frame: pd.DataFrame, macd_col: str = "macd", signal_col: str = "signal") -> pd.Series:
    """Return the current long/short state of MACD relative to its signal line."""
    if macd_col not in frame or signal_col not in frame:
        raise KeyError(f"Expected {macd_col!r} and {signal_col!r} columns")
    values = (frame[macd_col] > frame[signal_col]).astype("int8") - (frame[macd_col] < frame[signal_col]).astype("int8")
    valid = frame[macd_col].notna() & frame[signal_col].notna()
    return normalize_signal(values.where(valid, 0))


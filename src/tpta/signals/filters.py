"""Simple signal filters for momentum and trend regimes."""

from __future__ import annotations

import pandas as pd

from tpta.signals.base import normalize_signal


def rsi_filter(signal: pd.Series, rsi_values: pd.Series, lower: float = 30, upper: float = 70) -> pd.Series:
    """Suppress long signals above the upper bound and shorts below the lower bound."""
    if not 0 <= lower < upper <= 100:
        raise ValueError("RSI bounds must satisfy 0 <= lower < upper <= 100")
    signal, rsi_values = signal.align(rsi_values, join="left")
    return normalize_signal(signal.where(~(((signal > 0) & (rsi_values > upper)) | ((signal < 0) & (rsi_values < lower))), 0))


def trend_filter(signal: pd.Series, adx_values: pd.Series, minimum: float = 20) -> pd.Series:
    """Keep signals only when trend strength reaches the requested ADX level."""
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    signal, adx_values = signal.align(adx_values, join="left")
    return normalize_signal(signal.where(adx_values >= minimum, 0))


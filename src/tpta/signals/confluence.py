"""Combine independent directional signals."""

from __future__ import annotations

import pandas as pd

from tpta.signals.base import normalize_signal


def combine(signals: list[pd.Series], mode: str = "all", weights: list[float] | None = None) -> pd.Series:
    """Combine states using all, any, or weighted-majority agreement."""
    if not signals:
        raise ValueError("at least one signal is required")
    frame = pd.concat([normalize_signal(signal) for signal in signals], axis=1).fillna(0)
    if mode == "all":
        first = frame.iloc[:, 0]
        result = first.where(frame.eq(first, axis="index").all(axis=1), 0)
        result = result.where(frame.ne(0).all(axis=1), 0)
    elif mode == "any":
        total = frame.sum(axis=1)
        result = total.gt(0).astype(int) - total.lt(0).astype(int)
    elif mode == "majority":
        if weights is None:
            weights = [1.0] * len(signals)
        if len(weights) != len(signals):
            raise ValueError("weights must match the number of signals")
        total = frame.mul(weights, axis=1).sum(axis=1)
        result = total.gt(0).astype(int) - total.lt(0).astype(int)
    else:
        raise ValueError("mode must be all, any, or majority")
    return normalize_signal(result)


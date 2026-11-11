"""Simple equal-risk portfolio aggregation."""

from __future__ import annotations

import pandas as pd


def combine_returns(returns: dict[str, pd.Series], weights: dict[str, float] | None = None) -> pd.Series:
    """Combine instrument returns with normalized non-negative weights."""
    if not returns:
        raise ValueError("at least one return series is required")
    frame = pd.concat(returns, axis=1).fillna(0)
    weights = weights or {name: 1 / len(returns) for name in returns}
    if set(weights) != set(returns) or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must cover all assets and be non-negative with positive sum")
    normalized = pd.Series(weights, dtype=float) / sum(weights.values())
    return frame.mul(normalized, axis=1).sum(axis=1).rename("portfolio_return")


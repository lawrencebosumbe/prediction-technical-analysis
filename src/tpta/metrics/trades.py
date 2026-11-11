"""Trade-level performance metrics."""

from __future__ import annotations

import pandas as pd


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of closed trades with positive return."""
    values = trade_returns.dropna()
    return float(values.gt(0).mean()) if len(values) else 0.0


def profit_factor(trade_returns: pd.Series) -> float:
    """Gross winning return divided by gross losing return."""
    values = trade_returns.dropna()
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def expectancy(trade_returns: pd.Series) -> float:
    """Mean return per closed trade."""
    values = trade_returns.dropna()
    return float(values.mean()) if len(values) else 0.0


"""Return-based performance statistics."""

from __future__ import annotations

import math

import pandas as pd

from tpta.metrics.risk import max_drawdown, sharpe_ratio, sortino_ratio
from tpta.metrics.trades import profit_factor, win_rate


def summary(result: object, periods_per_year: int = 252) -> dict[str, float | int]:
    """Summarize a BacktestResult or a series of periodic returns."""
    returns = result.returns if hasattr(result, "returns") else result
    if not isinstance(returns, pd.Series):
        raise TypeError("summary expects a BacktestResult or pandas Series")
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "trades": 0}
    equity = (1 + clean).cumprod()
    years = len(clean) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else math.nan
    trades = getattr(result, "trades", pd.DataFrame())
    trade_returns = trades["return"] if isinstance(trades, pd.DataFrame) and "return" in trades else pd.Series(dtype=float)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": cagr,
        "sharpe": sharpe_ratio(clean, periods_per_year),
        "sortino": sortino_ratio(clean, periods_per_year),
        "max_drawdown": max_drawdown(clean),
        "win_rate": win_rate(trade_returns),
        "profit_factor": profit_factor(trade_returns),
        "trades": int(len(trade_returns)),
    }


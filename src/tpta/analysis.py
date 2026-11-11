"""Regime-conditional diagnostics for a backtest result."""

from __future__ import annotations

import pandas as pd

from tpta.metrics.risk import max_drawdown, sharpe_ratio, sortino_ratio
from tpta.metrics.trades import expectancy, profit_factor, win_rate


def regime_breakdown(
    result: object,
    regime_col: str,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize return and trade metrics by a regime column on the result's prices."""
    prices = getattr(result, "prices", None)
    returns = getattr(result, "returns", None)
    trades = getattr(result, "trades", pd.DataFrame())
    if not isinstance(prices, pd.DataFrame) or regime_col not in prices:
        raise ValueError(f"Backtest result must contain a prices column named {regime_col!r}")
    if not isinstance(returns, pd.Series):
        raise ValueError("backtest result must contain a return series")
    requested = metrics or ["sharpe", "win_rate", "profit_factor", "trades"]
    supported = {"sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "expectancy", "trades"}
    unknown = set(requested) - supported
    if unknown:
        raise ValueError(f"unsupported metric(s): {', '.join(sorted(unknown))}")

    aligned_regimes = prices[regime_col].reindex(returns.index)
    trades = trades.copy()
    if len(trades) and "entry_date" in trades:
        trades["regime"] = prices[regime_col].reindex(pd.DatetimeIndex(trades["entry_date"])).to_numpy()
    rows: dict[object, dict[str, float | int]] = {}
    for label, indices in aligned_regimes.groupby(aligned_regimes, dropna=True).groups.items():
        subset = returns.loc[indices].dropna()
        trade_returns = trades.loc[trades["regime"] == label, "return"] if "regime" in trades else pd.Series(dtype=float)
        functions = {
            "sharpe": lambda: sharpe_ratio(subset),
            "sortino": lambda: sortino_ratio(subset),
            "max_drawdown": lambda: max_drawdown(subset),
            "win_rate": lambda: win_rate(trade_returns),
            "profit_factor": lambda: profit_factor(trade_returns),
            "expectancy": lambda: expectancy(trade_returns),
            "trades": lambda: int(len(trade_returns)),
        }
        rows[label] = {metric: functions[metric]() for metric in requested}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(regime_col)


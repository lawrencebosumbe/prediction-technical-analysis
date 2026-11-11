"""Risk and risk-adjusted performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio with a zero risk-free rate."""
    clean = returns.dropna().astype(float)
    deviation = clean.std(ddof=1)
    if len(clean) < 2 or not np.isfinite(deviation) or deviation == 0:
        return 0.0
    return float(clean.mean() / deviation * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio using zero as the minimum acceptable return."""
    clean = returns.dropna().astype(float)
    downside = clean.clip(upper=0).pow(2).mean() ** 0.5
    if clean.empty or downside == 0:
        return 0.0
    return float(clean.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough drawdown in a return series."""
    equity = (1 + returns.fillna(0).astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized compound growth divided by absolute maximum drawdown."""
    clean = returns.dropna().astype(float)
    years = len(clean) / periods_per_year
    drawdown = abs(max_drawdown(clean))
    if not len(clean) or years <= 0 or drawdown == 0:
        return 0.0
    growth = (1 + clean).prod() ** (1 / years) - 1
    return float(growth / drawdown)


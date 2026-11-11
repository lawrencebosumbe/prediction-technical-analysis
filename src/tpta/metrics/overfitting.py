"""Lightweight diagnostics for parameter searches."""

from __future__ import annotations

import numpy as np
import pandas as pd


def deflated_sharpe_approx(returns: pd.Series, trials: int = 1, periods_per_year: int = 252) -> float:
    """Conservative Sharpe adjustment for the number of tried strategies."""
    if trials < 1:
        raise ValueError("trials must be positive")
    clean = returns.dropna().astype(float)
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return 0.0
    sharpe = clean.mean() / clean.std(ddof=1) * np.sqrt(periods_per_year)
    penalty = np.sqrt(2 * np.log(max(trials, 1)) / len(clean))
    return float(sharpe - penalty)


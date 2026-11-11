"""Out-of-sample strategy evaluation helpers."""

from __future__ import annotations

import pandas as pd

from tpta.model.walk_forward import WalkForward


def walk_forward_returns(returns: pd.Series, splitter: WalkForward) -> pd.Series:
    """Select test-only returns for each chronological fold."""
    parts = [returns.iloc[test] for _, test in splitter.split(len(returns))]
    if not parts:
        return returns.iloc[:0].rename("out_of_sample_return")
    return pd.concat(parts).rename("out_of_sample_return")


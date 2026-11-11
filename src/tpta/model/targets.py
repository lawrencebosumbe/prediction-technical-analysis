"""Forward return targets used only as labels, never as features."""

from __future__ import annotations

import pandas as pd


def forward_return_target(close: pd.Series, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """Label whether the close-to-close return over the next `horizon` bars is above threshold."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    future_return = close.shift(-horizon) / close - 1
    return future_return.gt(threshold).astype("Int8").where(future_return.notna()).rename(f"target_{horizon}")


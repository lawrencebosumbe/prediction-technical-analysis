"""Helpers for aligning multiple instrument series on a shared calendar."""

from __future__ import annotations

import pandas as pd


def align_prices(prices: dict[str, pd.DataFrame], column: str = "close", method: str | None = None) -> pd.DataFrame:
    """Align a mapping of symbol frames into a timestamp-by-symbol matrix."""
    if not prices:
        raise ValueError("at least one price frame is required")
    series = {symbol: frame[column].rename(symbol) for symbol, frame in prices.items()}
    aligned = pd.concat(series.values(), axis=1).sort_index()
    if method:
        aligned = aligned.ffill() if method == "ffill" else aligned
        if method not in {"ffill", "none"}:
            raise ValueError("method must be 'ffill' or 'none'")
    return aligned


"""Readable metric and correlation tables."""

from __future__ import annotations

import pandas as pd


def metrics_table(results: dict[str, dict[str, float | int]]) -> pd.DataFrame:
    """Create an asset-by-metric DataFrame."""
    return pd.DataFrame.from_dict(results, orient="index").rename_axis("symbol")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlations for aligned returns."""
    return returns.corr(min_periods=2)


"""Signal contracts shared by strategy rules."""

from __future__ import annotations

import pandas as pd


def normalize_signal(signal: pd.Series) -> pd.Series:
    """Normalize directional inputs to integer -1, 0, or +1 values."""
    values = pd.to_numeric(signal, errors="coerce").fillna(0)
    return values.clip(-1, 1).round().astype("int8").rename("signal")


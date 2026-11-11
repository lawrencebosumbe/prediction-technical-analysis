"""Volume based indicators."""

from __future__ import annotations

import pandas as pd


def obv(frame: pd.DataFrame) -> pd.Series:
    """On Balance Volume; if volume is unavailable, returns an all-NaN series."""
    if "volume" not in frame:
        return pd.Series(float("nan"), index=frame.index, name="obv")
    signed = frame["volume"].fillna(0) * frame["close"].diff().apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    return signed.cumsum().rename("obv")


def vwap(frame: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP for a single supplied session or data window."""
    if "volume" not in frame:
        raise ValueError("VWAP requires volume data")
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    volume = frame["volume"].fillna(0)
    denominator = volume.cumsum().replace(0, float("nan"))
    return ((typical * volume).cumsum() / denominator).rename("vwap")


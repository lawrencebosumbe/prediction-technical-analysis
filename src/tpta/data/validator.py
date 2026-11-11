"""Validation rules for canonical OHLCV market data."""

from __future__ import annotations

import pandas as pd


class DataIntegrityError(ValueError):
    """Raised when a market data frame violates the OHLCV contract."""


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a copy of an OHLCV frame with a UTC datetime index."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataIntegrityError("index must be a DatetimeIndex")
    if frame.empty:
        raise DataIntegrityError("price data is empty")

    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(result.columns)
    if missing:
        raise DataIntegrityError(f"missing required columns: {', '.join(sorted(missing))}")

    if result.index.hasnans:
        raise DataIntegrityError("index contains invalid timestamps")
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    if result.index.has_duplicates:
        raise DataIntegrityError("index contains duplicate timestamps")
    if not result.index.is_monotonic_increasing:
        raise DataIntegrityError("index must be sorted in ascending order")

    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise DataIntegrityError(f"{column} contains missing or non-numeric values")
        if (result[column] <= 0).any():
            raise DataIntegrityError(f"{column} must be positive")

    invalid = (result["high"] < result[["open", "low", "close"]].max(axis=1)) | (
        result["low"] > result[["open", "high", "close"]].min(axis=1)
    )
    if invalid.any():
        raise DataIntegrityError(f"OHLC bounds are inconsistent on {int(invalid.sum())} row(s)")

    if "volume" in result:
        result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
        if (result["volume"].dropna() < 0).any():
            raise DataIntegrityError("volume cannot be negative")
    return result


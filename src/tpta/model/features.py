"""Leakage-aware feature engineering."""

from __future__ import annotations

import pandas as pd

from tpta.indicators.macd import macd
from tpta.indicators.momentum import rsi
from tpta.indicators.moving_averages import ema, sma
from tpta.indicators.volatility import atr


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build features from present and past bars only."""
    close = prices["close"].astype(float)
    result = pd.DataFrame(index=prices.index)
    result["return_1"] = close.pct_change()
    result["return_5"] = close.pct_change(5)
    result["sma_gap_20"] = close / sma(close, 20) - 1
    result["sma_gap_50"] = close / sma(close, 50) - 1
    result["ema_gap_12"] = close / ema(close, 12) - 1
    result["rsi_14"] = rsi(close, 14) / 100
    result["atr_pct"] = atr(prices, 14) / close
    result["macd_pct"] = macd(close)["macd"] / close
    return result.replace([float("inf"), -float("inf")], float("nan"))


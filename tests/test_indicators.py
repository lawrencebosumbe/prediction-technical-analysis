import numpy as np
import pandas as pd

from tpta.indicators.macd import macd
from tpta.indicators.momentum import rsi
from tpta.indicators.moving_averages import ema, sma, wma
from tpta.indicators.volatility import atr


def test_moving_average_reference_values():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert sma(values, 2).tolist()[1:] == [1.5, 2.5, 3.5]
    assert wma(values, 2).tolist()[1:] == [5 / 3, 8 / 3, 11 / 3]
    assert np.isclose(ema(values, 2).iloc[1], 5 / 3)


def test_macd_and_rsi_are_finite_after_warmup():
    close = pd.Series(np.linspace(100, 120, 80))
    result = macd(close, fast=3, slow=6, signal=2)
    assert result.columns.tolist() == ["macd", "signal", "histogram"]
    assert result.iloc[-1].notna().all()
    assert rsi(close, 5).dropna().between(0, 100).all()


def test_atr_matches_constant_range():
    frame = pd.DataFrame({"high": [11.0] * 5, "low": [9.0] * 5, "close": [10.0] * 5})
    assert atr(frame, 2).dropna().eq(2.0).all()


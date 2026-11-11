import numpy as np
import pandas as pd

from tpta.indicators.moving_averages import ema, sma
from tpta.model.features import make_features


def test_indicators_are_prefix_stable():
    values = pd.Series(np.arange(1.0, 101.0))
    for indicator in (lambda x: sma(x, 10), lambda x: ema(x, 10)):
        full = indicator(values)
        prefix = indicator(values.iloc[:70])
        pd.testing.assert_series_equal(full.iloc[:70], prefix)


def test_features_use_only_observed_rows():
    index = pd.date_range("2024-01-01", periods=100, tz="UTC")
    close = pd.Series(np.linspace(100, 130, 100), index=index)
    frame = pd.DataFrame({"close": close, "open": close, "high": close + 1, "low": close - 1}, index=index)
    features = make_features(frame)
    truncated = make_features(frame.iloc[:70])
    pd.testing.assert_frame_equal(features.iloc[:70], truncated)


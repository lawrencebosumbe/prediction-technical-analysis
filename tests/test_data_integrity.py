from pathlib import Path

import pandas as pd
import pytest

from tpta.data.loader import read_csv_prices
from tpta.data.validator import DataIntegrityError, validate_ohlcv
from tpta.model.forecasts import SMAForecaster

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "data" / "raw").glob("*.csv")))
def test_bundled_csv_data_is_valid(path):
    frame = read_csv_prices(path)
    assert len(frame) > 10
    assert frame.index.is_monotonic_increasing
    assert str(frame.index.tz) == "UTC"
    assert {"open", "high", "low", "close"}.issubset(frame.columns)


def test_duplicate_timestamps_are_rejected():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    frame = pd.DataFrame({"open": [1, 1], "high": [2, 2], "low": [0.5, 0.5], "close": [1, 1]}, index=index)
    with pytest.raises(DataIntegrityError, match="duplicate"):
        validate_ohlcv(frame)


def test_sma_forecast_uses_only_the_trailing_window():
    index = pd.date_range("2024-01-01", periods=5, tz="UTC")
    forecast = SMAForecaster(window=3).forecast(pd.Series([1, 2, 3, 4, 5], index=index), steps=2)
    assert forecast.tolist() == [4.0, 4.0]
    assert forecast.index[0] > index[-1]


"""Small reference forecasters retained from the original analysis scripts."""

from __future__ import annotations

import pandas as pd

from tpta.data.calendar import future_sessions


class SMAForecaster:
    """Flat trailing-SMA projection, intended as a naive benchmark only."""

    def __init__(self, window: int = 10) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window

    def forecast(self, close: pd.Series, steps: int = 10, *, crypto: bool = False) -> pd.Series:
        """Project the latest trailing average onto future sessions."""
        values = close.dropna()
        if len(values) < self.window:
            raise ValueError(f"at least {self.window} close values are required")
        dates = future_sessions(pd.Timestamp(values.index[-1]), steps, crypto=crypto)
        return pd.Series(float(values.tail(self.window).mean()), index=dates, name="forecast")


class ARIMAForecaster:
    """ARIMA reference forecast using the optional statsmodels dependency."""

    def __init__(self, order: tuple[int, int, int] = (5, 1, 0)) -> None:
        if len(order) != 3 or min(order) < 0:
            raise ValueError("order must contain three non-negative integers")
        self.order = order

    def forecast(self, close: pd.Series, steps: int = 10, *, crypto: bool = False) -> pd.Series:
        """Fit ARIMA to observed closes and return future-session estimates."""
        values = close.dropna().astype(float)
        if steps < 1 or len(values) < max(sum(self.order) + 2, 5):
            raise ValueError("steps must be positive and the history must be long enough for ARIMA")
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise RuntimeError("Install the optional ML extra: pip install '.[ml]'") from exc
        predicted = ARIMA(values, order=self.order).fit().forecast(steps=steps)
        dates = future_sessions(pd.Timestamp(values.index[-1]), steps, crypto=crypto)
        return pd.Series(predicted.to_numpy(), index=dates, name="forecast")


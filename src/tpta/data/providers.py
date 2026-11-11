"""Local and optional remote market data providers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tpta.data.loader import read_csv_prices


class CSVProvider:
    """Load bundled or user-supplied CSV price data."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            project_data = Path(__file__).resolve().parents[3] / "data" / "raw"
            package_data = Path(__file__).resolve().parents[1] / "resources" / "raw"
            self.data_dir = project_data if project_data.is_dir() else package_data

    def load(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        path = self.data_dir / f"{symbol.lower()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No CSV data for {symbol!r}; expected {path}")
        frame = read_csv_prices(path)
        return _date_filter(frame, start, end)


class YFinanceProvider:
    """Fetch prices through the optional yfinance package."""

    def load(
        self, symbol: str, start: str | None = None, end: str | None = None, interval: str = "1d"
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install the optional data extra: pip install '.[data]'") from exc
        ticker = yf.Ticker(symbol)
        raw = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        if raw.empty:
            raise ValueError(f"Provider returned no prices for {symbol!r}")
        raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")
        raw = raw.rename(columns={"Adj Close": "close", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        return raw[[column for column in ("open", "high", "low", "close", "volume") if column in raw]].rename(
            columns=str.lower
        )


def _date_filter(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        boundary = pd.Timestamp(start)
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
        frame = frame.loc[frame.index >= boundary]
    if end:
        boundary = pd.Timestamp(end)
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
        frame = frame.loc[frame.index < boundary]
    if frame.empty:
        raise ValueError("No rows remain in the requested date range")
    return frame


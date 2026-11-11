"""Load historical prices from CSV files or configured providers."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from tpta.data.validator import validate_ohlcv


def _number(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "-", "--"}:
        return float("nan")
    if text.endswith("%"):
        text = text[:-1]
    multiplier = 1.0
    if text and text[-1].upper() in {"K", "M", "B"}:
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[text[-1].upper()]
        text = text[:-1]
    return float(text) * multiplier


def read_csv_prices(path: str | Path) -> pd.DataFrame:
    """Read common financial-download CSV formats into canonical OHLCV columns."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8-sig")
    try:
        raw = pd.read_csv(io.StringIO(raw_text))
        if len(raw.columns) == 1:
            raise ValueError("single-column export")
    except (ValueError, pd.errors.ParserError):
        # Some exports wrap each complete row in quotes and double every inner quote.
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        repaired = "\n".join(line[1:-1].replace('""', '"') if line.startswith('"') and line.endswith('"') else line for line in lines)
        raw = pd.read_csv(io.StringIO(repaired))

    raw.columns = [str(column).strip().strip('"').lower() for column in raw.columns]
    aliases = {"price": "close", "adj close": "close", "date/time": "date", "vol.": "volume", "vol": "volume"}
    raw = raw.rename(columns=aliases)
    if "date" not in raw:
        raise ValueError(f"{path} must have a Date column")
    if "close" not in raw and "last" in raw:
        raw = raw.rename(columns={"last": "close"})
    if "close" not in raw:
        raise ValueError(f"{path} must have a Price/Close column")

    dates = pd.to_datetime(raw.pop("date"), errors="coerce", format="mixed", dayfirst=False, utc=True)
    frame = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    frame["close"] = raw["close"].map(_number).to_numpy()
    for column in ("open", "high", "low"):
        frame[column] = raw[column].map(_number).to_numpy() if column in raw else frame["close"]
    if "volume" in raw:
        frame["volume"] = raw["volume"].map(_number).to_numpy()
    frame = frame.loc[~frame.index.isna()].sort_index()
    # Duplicate dates are not silently discarded: the validator reports them.
    return validate_ohlcv(frame)


def load_prices(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    provider: str = "csv",
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load a symbol's OHLCV history. The bundled CSV provider works offline."""
    from tpta.data.providers import CSVProvider, YFinanceProvider

    if provider == "csv":
        result = CSVProvider(data_dir).load(symbol, start, end)
    elif provider == "yfinance":
        result = YFinanceProvider().load(symbol, start, end, interval)
    else:
        raise ValueError(f"Unsupported provider {provider!r}; choose csv or yfinance")
    return validate_ohlcv(result)


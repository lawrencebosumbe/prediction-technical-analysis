# Methodology

This repository is a research toolkit. Its examples do not establish a trading edge and do not constitute financial advice.

## Data and time

Input CSVs are normalized to `open`, `high`, `low`, `close`, and optional `volume` columns on a unique, increasing UTC `DatetimeIndex`. Dates in the bundled downloads are interpreted month first. Invalid dates are dropped; duplicate timestamps and invalid OHLC ranges raise `DataIntegrityError`.

## Indicators and labels

Indicators use the current bar and earlier bars only. Rolling indicators return missing values during their warm-up period. `forward_return_target` intentionally looks ahead to create labels; it must never be included in the feature matrix.

## Signal timing and backtests

Signal states are shifted one bar before exposure is applied. The baseline engine applies signals at the next bar's open when available, deducts transaction costs when exposure changes, and charges financing while positions are open. Optional ATR stops and targets use OHLC bars; when both levels are crossed in one bar, the stop is assumed to fill first. Results are illustrative and omit instrument-specific contract multipliers, margin rules, and exchange holidays.

## Reproducibility

Use the bundled CSV provider for offline repeatability. Store fetched data in `data/cache` and record source, date range, config, and package version for every research run. Evaluate parameter choices on chronological out-of-sample periods.


# FAQ

## Does the project forecast prices?

The baseline indicators and logistic regression helper estimate historical patterns. They do not guarantee a future outcome. Flat moving-average extrapolation from the original scripts is not presented as a predictive model.

## Can I run it without a network connection?

Yes. Use `provider: csv` (the default) with the bundled histories. `yfinance` is optional and requires installing the `data` extra.

## Where are reports written?

The CLI writes PNG charts and a small HTML report to the `evaluation.output_dir` in a config. Generated reports are ignored by Git.

## Why are sample backtests short?

The supplied CSV files cover only a small historical window. Results from them are demonstrations of the software path, not strategy evidence.


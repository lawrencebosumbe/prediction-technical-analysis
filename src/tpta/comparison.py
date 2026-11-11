"""Cross-instrument baseline strategy comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tpta.backtest import Backtester, CostModel
from tpta.data import load_prices
from tpta.indicators import macd, sma
from tpta.metrics import summary
from tpta.signals import combine, macd_crossover, sma_crossover


@dataclass
class ComparisonResult:
    """Per-symbol backtest results and comparison visualizations."""

    results: dict[str, object]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({symbol: summary(result) for symbol, result in self.results.items()}).T

    def plot_equity(self, output: str | Path | None = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        for symbol, result in self.results.items():
            relative = result.equity / result.equity.iloc[0]
            relative.plot(ax=ax, label=symbol)
        ax.set(title="Cross-market equity curves", ylabel="Growth of 1", xlabel="Date")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        if output:
            fig.savefig(output, dpi=150)
        return fig

    def plot_heatmap(self, output: str | Path | None = None):
        table = self.table().select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(max(7, len(table.columns)), max(3, len(table) * 0.5)))
        image = ax.imshow(table.fillna(0).to_numpy(), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(table.index)), table.index)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        if output:
            fig.savefig(output, dpi=150)
        return fig


def compare(
    symbols: list[str],
    strategy: str = "sma_macd",
    start: str | None = None,
    end: str | None = None,
    output: str | Path | None = None,
) -> ComparisonResult:
    """Run the same short/long SMA plus MACD baseline across symbols."""
    if not symbols:
        raise ValueError("symbols cannot be empty")
    if strategy not in {"sma_macd", "sma_macd_confluence"}:
        raise ValueError("supported strategy is sma_macd")
    results: dict[str, object] = {}
    for symbol in symbols:
        prices = load_prices(symbol, start=start, end=end)
        prices["sma_fast"] = sma(prices.close, 5)
        prices["sma_slow"] = sma(prices.close, 15)
        prices = prices.join(macd(prices.close))
        signal = combine([sma_crossover(prices), macd_crossover(prices)], mode="any")
        results[symbol] = Backtester(prices, signal, CostModel()).run()
    result = ComparisonResult(results)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.plot_equity(output.with_name(output.stem + "_equity.png"))
        result.plot_heatmap(output.with_name(output.stem + "_heatmap.png"))
        output.write_text(result.table().to_html(), encoding="utf-8")
    return result


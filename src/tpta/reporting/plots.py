"""Matplotlib charts using a headless-safe backend by default."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(equity: pd.Series, output: str | Path | None = None, show: bool = False):
    """Plot an equity curve and optionally save it to disk."""
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax, color="#1769aa", linewidth=1.6)
    ax.set(title="Strategy equity", xlabel="Date", ylabel="Equity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_price_indicators(prices: pd.DataFrame, output: str | Path | None = None, show: bool = False):
    """Plot close and any SMA/MACD columns present in the frame."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    prices["close"].plot(ax=axes[0], label="Close", color="#222222")
    for column in ("sma_fast", "sma_slow"):
        if column in prices:
            prices[column].plot(ax=axes[0], label=column)
    axes[0].legend()
    axes[0].set_ylabel("Price")
    if "macd" in prices:
        prices["macd"].plot(ax=axes[1], label="MACD")
    if "signal" in prices:
        prices["signal"].plot(ax=axes[1], label="Signal")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].legend()
    axes[1].set_ylabel("MACD")
    fig.tight_layout()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


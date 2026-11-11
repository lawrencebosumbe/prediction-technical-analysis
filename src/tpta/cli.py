"""Command-line interface for local analysis and backtests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tpta.backtest import Backtester, CostModel
from tpta.data import load_prices
from tpta.indicators import macd, sma
from tpta.metrics import summary
from tpta.reporting.html import write_html_report
from tpta.reporting.plots import plot_equity, plot_price_indicators
from tpta.signals import combine, macd_crossover, sma_crossover
from tpta.comparison import compare

INSTRUMENTS = {
    "US30": "Dow Jones Industrial Average",
    "US500": "S&P 500",
    "US100": "Nasdaq 100",
    "CAC40": "CAC 40",
    "CHINA50": "FTSE China A50",
    "DAX30": "DAX",
    "FTSE100": "FTSE 100",
    "NIKKEI225": "Nikkei 225",
    "STOXX50E": "Euro Stoxx 50",
    "BTCUSD": "Bitcoin / US Dollar",
}


def _project_root() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    return source_root if (source_root / "configs").is_dir() else Path.cwd()


def _read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _build_signals(prices: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    frame = prices.copy()
    indicator_config = config.get("indicators", {})
    sma_config = indicator_config.get("sma", [])
    windows = [int(item["window"] if isinstance(item, dict) else item) for item in sma_config]
    rules = config.get("signals", {}).get("rules", [])
    for rule in rules:
        if str(rule.get("type", "")).lower() == "sma_crossover":
            windows.extend(int(rule[key]) for key in ("fast", "slow") if str(rule.get(key, "")).isdigit())
    if not windows:
        windows = [20, 50]
    windows = sorted(set(windows))
    for window in windows:
        frame[f"sma_{window}"] = sma(frame["close"], window)

    macd_config = indicator_config.get("macd", {})
    macd_frame = macd(
        frame["close"],
        fast=int(macd_config.get("fast", 12)),
        slow=int(macd_config.get("slow", 26)),
        signal=int(macd_config.get("signal", 9)),
    )
    frame = frame.join(macd_frame)

    signal_parts: list[pd.Series] = []
    for rule in rules:
        kind = str(rule.get("type", "")).lower()
        if kind == "sma_crossover":
            fast, slow = int(rule.get("fast", windows[0])), int(rule.get("slow", windows[-1]))
            signal_parts.append(sma_crossover(frame, f"sma_{fast}", f"sma_{slow}"))
        elif kind == "macd_crossover":
            signal_parts.append(macd_crossover(frame))
        else:
            raise ValueError(f"Unsupported signal rule type: {kind!r}")
    if not signal_parts:
        signal_parts = [sma_crossover(frame, f"sma_{windows[0]}", f"sma_{windows[-1]}")]
    combination = config.get("signals", {}).get("combination", "all")
    return frame, combine(signal_parts, mode=combination)


def run_backtest_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run configured symbol backtests and persist chart/report artifacts."""
    root = _project_root()
    data_config = config.get("data", {})
    symbols = data_config.get("symbols", [data_config.get("symbol", "US500")])
    if isinstance(symbols, str):
        symbols = [part.strip() for part in symbols.split(",") if part.strip()]
    if not symbols:
        raise ValueError("data.symbols must contain at least one instrument")
    execution = config.get("execution", {})
    cost_data = config.get("costs", {})
    cost_model = CostModel(
        spread_points=float(cost_data.get("spread_points", 0)),
        commission_per_lot=float(cost_data.get("commission_per_lot", 0)),
        slippage_bps=float(cost_data.get("slippage_bps", 0)),
        financing_daily_bps=float(cost_data.get("financing_daily_bps", 0)),
        financing_on_weekends=bool(cost_data.get("financing_on_weekends", True)),
    )
    output_dir = Path(config.get("evaluation", {}).get("output_dir", "reports"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    results: dict[str, Any] = {}
    for symbol in symbols:
        prices = load_prices(
            str(symbol),
            start=data_config.get("start"),
            end=data_config.get("end"),
            interval=data_config.get("interval", "1d"),
            provider=data_config.get("provider", "csv"),
        )
        frame, signal = _build_signals(prices, config)
        result = Backtester(
            prices=frame,
            signal=signal,
            cost_model=cost_model,
            initial_capital=float(execution.get("initial_capital", 100_000)),
            risk_per_trade=float(execution.get("risk_per_trade", 0.01)),
            stop_loss_atr=execution.get("stop_loss_atr"),
            take_profit_atr=execution.get("take_profit_atr"),
            allow_short=bool(execution.get("allow_short", True)),
        ).run()
        metrics = summary(result)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure = plot_price_indicators(frame, output_dir / f"{symbol.lower()}_analysis.png")
        plt.close(figure)
        figure = plot_equity(result.equity, output_dir / f"{symbol.lower()}_equity.png")
        plt.close(figure)
        write_html_report(metrics, result.equity, output_dir / f"{symbol.lower()}_report.html")
        results[str(symbol)] = {"metrics": metrics, "result": result}
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tpta", description="Technical analysis and backtest research toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("instruments", help="list bundled sample instruments")
    backtest = commands.add_parser("backtest", help="run a backtest from YAML configuration")
    backtest.add_argument("--config", required=True)
    fetch = commands.add_parser("fetch", help="load prices and save a local CSV copy")
    fetch.add_argument("--symbols", required=True, help="comma-separated symbols")
    fetch.add_argument("--start")
    fetch.add_argument("--end")
    fetch.add_argument("--interval", default="1d")
    fetch.add_argument("--provider", choices=["csv", "yfinance"], default="csv")
    fetch.add_argument("--output-dir", default="data/cache")
    analyze = commands.add_parser("analyze", help="save price, SMA, and MACD charts for one symbol")
    analyze.add_argument("--symbol", required=True)
    analyze.add_argument("--sma-fast", type=int, default=20)
    analyze.add_argument("--sma-slow", type=int, default=50)
    analyze.add_argument("--output-dir", default="reports")
    comparison = commands.add_parser("compare", help="compare the baseline across symbols")
    comparison.add_argument("--symbols", required=True, help="comma-separated symbols")
    comparison.add_argument("--strategy", default="sma_macd")
    comparison.add_argument("--output", default="reports/comparison.html")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the TPTA command-line interface."""
    args = _parser().parse_args(argv)
    root = _project_root()
    if args.command == "instruments":
        for symbol, name in INSTRUMENTS.items():
            available = (root / "data" / "raw" / f"{symbol.lower()}.csv").exists() or (
                Path(__file__).resolve().parent / "resources" / "raw" / f"{symbol.lower()}.csv"
            ).exists()
            print(f"{symbol:10} {name}{'' if available else ' (no bundled data)'}")
        return 0
    if args.command == "backtest":
        config_path = Path(args.config)
        config = _read_config(config_path if config_path.is_absolute() else root / config_path)
        results = run_backtest_config(config)
        serializable = {
            symbol: {
                key: (value if not isinstance(value, float) or math.isfinite(value) else None)
                for key, value in item["metrics"].items()
            }
            for symbol, item in results.items()
        }
        print(json.dumps(serializable, indent=2, allow_nan=False))
        return 0
    if args.command == "fetch":
        destination = Path(args.output_dir)
        if not destination.is_absolute():
            destination = root / destination
        destination.mkdir(parents=True, exist_ok=True)
        for symbol in (part.strip() for part in args.symbols.split(",")):
            prices = load_prices(symbol, args.start, args.end, args.interval, args.provider)
            prices.to_csv(destination / f"{symbol.lower()}.csv", index_label="date")
            print(f"saved {len(prices)} rows for {symbol} to {destination}")
        return 0
    if args.command == "analyze":
        prices = load_prices(args.symbol)
        prices["sma_fast"] = sma(prices["close"], args.sma_fast)
        prices["sma_slow"] = sma(prices["close"], args.sma_slow)
        prices = prices.join(macd(prices["close"]))
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = root / out
        path = out / f"{args.symbol.lower()}_analysis.png"
        figure = plot_price_indicators(prices, path)
        plt.close(figure)
        print(path)
        return 0
    if args.command == "compare":
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        symbols = list(INSTRUMENTS) if args.symbols.lower() == "all" else [
            part.strip() for part in args.symbols.split(",") if part.strip()
        ]
        result = compare(symbols, strategy=args.strategy, output=output)
        print(result.table().to_string())
        print(output)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


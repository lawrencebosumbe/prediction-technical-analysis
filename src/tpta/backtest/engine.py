"""Next-bar, close-to-close strategy simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tpta.backtest.costs import CostModel
from tpta.backtest.sizing import fractional_exposure
from tpta.indicators.volatility import atr
from tpta.signals.base import normalize_signal


@dataclass
class BacktestResult:
    """Equity and diagnostics from one backtest run."""

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    prices: pd.DataFrame | None = None

    def plot(self, output: str | None = None, show: bool = False):
        """Plot the resulting equity curve."""
        from tpta.reporting.plots import plot_equity

        return plot_equity(self.equity, output, show)

    def report(self, output: str | None = None):
        """Write an HTML metrics and equity report."""
        from tpta.metrics.returns import summary
        from tpta.reporting.html import write_html_report

        return write_html_report(summary(self), self.equity, output or "reports/backtest.html")


class Backtester:
    """Simulate a directional signal with next-bar execution and explicit costs."""

    def __init__(
        self,
        prices: pd.DataFrame,
        signal: pd.Series,
        cost_model: CostModel | None = None,
        initial_capital: float = 100_000.0,
        risk_per_trade: float = 0.01,
        stop_loss_atr: float | None = None,
        take_profit_atr: float | None = None,
        allow_short: bool = True,
    ) -> None:
        if initial_capital <= 0 or not 0 < risk_per_trade <= 1:
            raise ValueError("initial_capital must be positive and risk_per_trade in (0, 1]")
        if stop_loss_atr is not None and stop_loss_atr <= 0:
            raise ValueError("stop_loss_atr must be positive")
        if take_profit_atr is not None and take_profit_atr <= 0:
            raise ValueError("take_profit_atr must be positive")
        self.prices = prices.copy()
        self.signal = normalize_signal(signal.reindex(prices.index).fillna(0))
        if not allow_short:
            self.signal = self.signal.clip(lower=0)
        self.cost_model = cost_model or CostModel()
        self.initial_capital = float(initial_capital)
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr

    def run(self) -> BacktestResult:
        """Run with next-bar execution, spread/slippage, financing, and optional ATR exits."""
        if self.prices.empty:
            raise ValueError("prices cannot be empty")
        close = self.prices["close"].astype(float)
        opens = self.prices["open"].astype(float) if "open" in self.prices else close.shift(1).fillna(close)
        high = self.prices["high"].astype(float) if "high" in self.prices else close
        low = self.prices["low"].astype(float) if "low" in self.prices else close
        atr_values = atr(self.prices, 14).shift(1) if self.stop_loss_atr or self.take_profit_atr else pd.Series(float("nan"), index=close.index)

        active = 0
        exposure = 0.0
        stop = target = None
        daily_positions: list[float] = []
        daily_returns: list[float] = []
        trade_rows: list[dict[str, object]] = []
        entry_date = None
        entry_price = 0.0
        trade_direction = 0
        previous_close = float(close.iloc[0])
        for i, timestamp in enumerate(close.index):
            bar_open, bar_close = float(opens.iloc[i]), float(close.iloc[i])
            overnight = active * exposure * (bar_open / previous_close - 1) if i else 0.0
            requested = int(self.signal.iloc[i - 1]) if i else 0
            current_atr = float(atr_values.iloc[i]) if pd.notna(atr_values.iloc[i]) else float("nan")
            if requested != active and (self.stop_loss_atr or self.take_profit_atr) and not (current_atr > 0):
                requested = active
            turnover = abs(requested - active) * exposure if requested != active else 0.0
            if requested != active:
                if active and entry_date is not None:
                    trade_rows.append({"entry_date": entry_date, "exit_date": timestamp, "direction": trade_direction, "entry_price": entry_price, "exit_price": bar_open, "return": trade_direction * (bar_open / entry_price - 1)})
                active = requested
                stop = target = None
                exposure = 0.0
                entry_date, entry_price = None, 0.0
                if active:
                    if self.stop_loss_atr and current_atr > 0:
                        distance_fraction = current_atr * self.stop_loss_atr / bar_open
                        exposure = fractional_exposure(self.risk_per_trade, distance_fraction)
                    else:
                        exposure = 1.0
                    if current_atr > 0:
                        stop = bar_open - active * self.stop_loss_atr * current_atr if self.stop_loss_atr else None
                        target = bar_open + active * self.take_profit_atr * current_atr if self.take_profit_atr else None
                    entry_date, entry_price, trade_direction = timestamp, bar_open, active

            intraday = 0.0
            stop_hit = target_hit = False
            if active:
                if stop is not None:
                    stop_hit = (float(low.iloc[i]) <= stop) if active > 0 else (float(high.iloc[i]) >= stop)
                    stop_hit = stop_hit or (bar_open <= stop if active > 0 else bar_open >= stop)
                if target is not None:
                    target_hit = (float(high.iloc[i]) >= target) if active > 0 else (float(low.iloc[i]) <= target)
                if stop_hit or target_hit:
                    # If both levels trade during the same OHLC bar, assume the adverse stop fills first.
                    gap_stop = stop is not None and (bar_open <= stop if active > 0 else bar_open >= stop)
                    exit_price = bar_open if gap_stop else stop if stop_hit else target
                    intraday = active * exposure * (float(exit_price) / bar_open - 1)
                    turnover += exposure
                    if entry_date is not None:
                        trade_rows.append({"entry_date": entry_date, "exit_date": timestamp, "direction": trade_direction, "entry_price": entry_price, "exit_price": float(exit_price), "return": trade_direction * (float(exit_price) / entry_price - 1)})
                    entry_date, entry_price = None, 0.0
                    active, exposure, stop, target = 0, 0.0, None, None
                else:
                    intraday = active * exposure * (bar_close / bar_open - 1)

            gross_return = (1 + overnight) * (1 + intraday) - 1
            cost = self.cost_model.transaction_fraction(bar_open, turnover, self.initial_capital)
            financing = exposure * self.cost_model.financing_daily_bps / 10_000
            if not self.cost_model.financing_on_weekends and timestamp.dayofweek >= 5:
                financing = 0.0
            daily_returns.append(gross_return - cost - financing)
            daily_positions.append(float(active) * exposure)
            previous_close = bar_close

        positions = pd.Series(daily_positions, index=close.index, name="position")
        net_returns = pd.Series(daily_returns, index=close.index, name="strategy_return")
        equity = (1 + net_returns).cumprod().mul(self.initial_capital).rename("equity")
        if active and entry_date is not None:
            last_date, last_price = close.index[-1], float(close.iloc[-1])
            trade_rows.append({"entry_date": entry_date, "exit_date": last_date, "direction": trade_direction, "entry_price": entry_price, "exit_price": last_price, "return": trade_direction * (last_price / entry_price - 1)})
        trades = pd.DataFrame(trade_rows, columns=["entry_date", "exit_date", "direction", "entry_price", "exit_price", "return"])
        return BacktestResult(
            equity=equity,
            returns=net_returns,
            positions=positions,
            trades=trades,
            prices=self.prices.copy(),
        )


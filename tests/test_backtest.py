import pandas as pd
import pytest

from tpta.backtest import Backtester, CostModel
from tpta.strategy import Strategy


def test_signal_is_executed_on_next_bar():
    index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    close = pd.Series([100.0, 110.0, 121.0], index=index)
    prices = pd.DataFrame({"close": close})
    signal = pd.Series([1, 1, 1], index=index)
    result = Backtester(prices, signal, initial_capital=1000, cost_model=CostModel()).run()
    assert result.positions.tolist() == [0.0, 1.0, 1.0]
    assert result.equity.tolist() == pytest.approx([1000.0, 1100.0, 1210.0])


def test_higher_transaction_cost_reduces_equity():
    index = pd.date_range("2024-01-01", periods=5, tz="UTC")
    close = pd.Series([100, 102, 101, 104, 103], index=index, dtype=float)
    prices = pd.DataFrame({"close": close})
    signal = pd.Series([1, 1, -1, -1, 0], index=index)
    low = Backtester(prices, signal, cost_model=CostModel()).run()
    high = Backtester(prices, signal, cost_model=CostModel(spread_points=2, slippage_bps=10)).run()
    assert high.equity.iloc[-1] < low.equity.iloc[-1]


def test_strategy_expression_and_short_entry():
    index = pd.date_range("2024-01-01", periods=4, tz="UTC")
    close = pd.Series([10.0, 11.0, 9.0, 8.0], index=index)
    prices = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close})
    strategy = Strategy(
        prices,
        indicators={"fast": lambda frame: frame.close, "previous": lambda frame: frame.close.shift(1)},
        entry="fast > previous",
        exit="fast < previous",
        allow_short=True,
    )
    result = strategy.backtest()
    assert result.positions.tolist() == [0.0, 0.0, 1.0, -1.0]


def test_atr_stop_uses_bar_range_and_records_stop_fill():
    index = pd.date_range("2024-01-01", periods=20, tz="UTC")
    close = pd.Series([100.0] * 20, index=index)
    low = pd.Series([99.0] * 20, index=index)
    low.iloc[14] = 97.0
    prices = pd.DataFrame({"open": close, "high": 101.0, "low": low, "close": close})
    signal = pd.Series([1] * 20, index=index)
    result = Backtester(prices, signal, risk_per_trade=0.01, stop_loss_atr=1.0).run()
    assert result.trades.iloc[0].exit_price == 98.0
    assert result.returns.iloc[14] == pytest.approx(-0.01)


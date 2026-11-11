import pandas as pd

from tpta.analysis import regime_breakdown
from tpta.backtest import Backtester
from tpta.metrics.returns import summary
from tpta.metrics.risk import max_drawdown, sharpe_ratio
from tpta.metrics.trades import profit_factor, win_rate


def test_known_answer_metrics():
    returns = pd.Series([0.1, -0.1, 0.1, -0.1])
    assert max_drawdown(returns) < 0
    assert sharpe_ratio(pd.Series([0.01, 0.02, 0.03])) > 0
    assert win_rate(pd.Series([0.1, -0.05, 0.02])) == 2 / 3
    assert profit_factor(pd.Series([0.1, -0.05, 0.02])) == 2.4
    assert summary(returns)["trades"] == 0


def test_regime_breakdown_uses_backtest_price_context():
    index = pd.date_range("2024-01-01", periods=4, tz="UTC")
    close = pd.Series([100.0, 101.0, 99.0, 100.0], index=index)
    prices = pd.DataFrame({"close": close, "regime": ["up", "up", "down", "down"]})
    result = Backtester(prices, pd.Series([1, 1, -1, -1], index=index)).run()
    table = regime_breakdown(result, "regime", ["sharpe", "trades"])
    assert set(table.index) == {"up", "down"}


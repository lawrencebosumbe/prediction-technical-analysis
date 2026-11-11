"""Prediction Technical Analysis toolkit."""

from tpta.data import load_prices
from tpta.indicators import macd, sma
from tpta.strategy import Strategy

__version__ = "0.1.0"

__all__ = ["Strategy", "compare", "load_prices", "macd", "sma", "__version__"]


def compare(*args, **kwargs):
    """Lazily run cross-instrument comparisons without choosing a plotting backend on import."""
    from tpta.comparison import compare as run_comparison

    return run_comparison(*args, **kwargs)


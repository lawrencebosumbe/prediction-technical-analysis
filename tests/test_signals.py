import pandas as pd

from tpta.signals import combine, macd_crossover, sma_crossover


def test_sma_crossover_returns_directional_state():
    frame = pd.DataFrame({"fast": [1, 2, 3, 2], "slow": [2, 2, 2, 3]})
    assert sma_crossover(frame, "fast", "slow").tolist() == [-1, 0, 1, -1]


def test_combine_modes():
    left = pd.Series([1, 1, -1, 0])
    right = pd.Series([1, -1, -1, 1])
    assert combine([left, right], "all").tolist() == [1, 0, -1, 0]
    assert combine([left, right], "any").tolist() == [1, 0, -1, 1]


def test_macd_crossover():
    frame = pd.DataFrame({"macd": [1, 0, -1], "signal": [0, 0, 1]})
    assert macd_crossover(frame).tolist() == [1, 0, -1]


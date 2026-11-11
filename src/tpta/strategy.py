"""Small expression-based strategy convenience wrapper."""

from __future__ import annotations

import re
from collections.abc import Callable

import pandas as pd

from tpta.backtest import BacktestResult, Backtester, CostModel

_COMPARISON = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*(>=|<=|==|!=|>|<)\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|-?\d+(?:\.\d+)?)$")


class Strategy:
    """Build indicator columns, evaluate simple comparisons, and backtest them."""

    def __init__(
        self,
        prices: pd.DataFrame,
        indicators: dict[str, Callable[[pd.DataFrame], pd.Series | pd.DataFrame]],
        entry: str,
        exit: str,
        allow_short: bool = False,
        short_entry: str | None = None,
        short_exit: str | None = None,
        cost_model: CostModel | None = None,
        initial_capital: float = 100_000.0,
        risk_per_trade: float = 0.01,
        stop_loss_atr: float | None = None,
        take_profit_atr: float | None = None,
    ) -> None:
        self.prices = prices.copy()
        self.indicators = indicators
        self.entry = entry
        self.exit = exit
        self.allow_short = allow_short
        self.short_entry = short_entry
        self.short_exit = short_exit
        self.cost_model = cost_model or CostModel()
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr

    def backtest(self) -> BacktestResult:
        """Evaluate simple `column comparator column-or-number` rule expressions."""
        frame = self.prices.copy()
        for name, function in self.indicators.items():
            values = function(frame)
            if isinstance(values, pd.DataFrame):
                for column in values:
                    frame[f"{name}_{column}"] = values[column]
                    if column not in frame:
                        frame[column] = values[column]
            else:
                frame[name] = values
        long_entry = _expression(self.entry, frame)
        exit_condition = _expression(self.exit, frame)
        short_entry = (
            _expression(self.short_entry or _mirror_expression(self.entry), frame)
            if self.allow_short
            else pd.Series(False, index=frame.index)
        )
        short_exit = (
            _expression(self.short_exit or _mirror_expression(self.exit), frame)
            if self.allow_short
            else pd.Series(False, index=frame.index)
        )
        state = 0
        values: list[int] = []
        for long_signal, short_signal, exit_signal, short_exit_signal in zip(long_entry, short_entry, exit_condition, short_exit):
            if (state == 1 and exit_signal) or (state == -1 and short_exit_signal):
                state = 0
            if long_signal:
                state = 1
            if self.allow_short and short_signal:
                state = -1
            values.append(state)
        signal = pd.Series(values, index=frame.index, name="signal")
        return Backtester(
            frame,
            signal,
            cost_model=self.cost_model,
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            stop_loss_atr=self.stop_loss_atr,
            take_profit_atr=self.take_profit_atr,
            allow_short=self.allow_short,
        ).run()


def _expression(expression: str, frame: pd.DataFrame) -> pd.Series:
    """Parse conjunction/disjunctions of comparisons without arbitrary code evaluation."""
    expression = expression.strip().strip("()")
    for operator, combine in ((" or ", lambda a, b: a | b), (" and ", lambda a, b: a & b)):
        clauses = expression.split(operator)
        if len(clauses) > 1:
            values = [_expression(clause, frame) for clause in clauses]
            result = values[0]
            for value in values[1:]:
                result = combine(result, value)
            return result
    match = _COMPARISON.fullmatch(expression)
    if not match:
        raise ValueError(f"Unsupported rule expression: {expression!r}")
    left, operator, right = match.groups()
    left = left.replace(".", "_")
    if left not in frame:
        raise KeyError(f"Rule references unknown column {left!r}")
    lhs = frame[left]
    right_name = right.replace(".", "_")
    rhs = frame[right_name] if right_name in frame else float(right)
    operations = {">": lhs.__gt__, "<": lhs.__lt__, ">=": lhs.__ge__, "<=": lhs.__le__, "==": lhs.__eq__, "!=": lhs.__ne__}
    return operations[operator](rhs).fillna(False)


def _mirror_expression(expression: str) -> str:
    """Reverse directional comparisons while keeping the rule's conjunctions."""
    flipped = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "==": "==", "!=": "!="}
    parts = re.split(r"\s+(and|or)\s+", expression.strip(), flags=re.IGNORECASE)
    mirrored: list[str] = []
    for part in parts:
        if part.lower() in {"and", "or"}:
            mirrored.append(part)
            continue
        match = _COMPARISON.fullmatch(part.strip().strip("()"))
        if not match:
            raise ValueError(f"Cannot infer opposite rule from {part!r}; pass short_entry explicitly")
        left, operator, right = match.groups()
        mirrored.append(f"{left} {flipped[operator]} {right}")
    return " ".join(mirrored)


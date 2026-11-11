"""Transparent proportional transaction and financing costs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Cost assumptions; spreads are points and slippage/financing are basis points."""

    spread_points: float = 0.0
    commission_per_lot: float = 0.0
    slippage_bps: float = 0.0
    financing_daily_bps: float = 0.0
    financing_on_weekends: bool = True

    def __post_init__(self) -> None:
        if min(self.spread_points, self.commission_per_lot, self.slippage_bps, self.financing_daily_bps) < 0:
            raise ValueError("cost assumptions cannot be negative")

    def transaction_fraction(self, price: float, turnover: float, capital: float = 1.0) -> float:
        """Return the equity fraction charged for a change in exposure."""
        if price <= 0 or capital <= 0:
            raise ValueError("price and capital must be positive")
        return abs(turnover) * (self.spread_points / price + self.slippage_bps / 10_000) + abs(turnover) * self.commission_per_lot / capital


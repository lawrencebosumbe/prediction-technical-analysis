"""Position sizing helpers."""

from __future__ import annotations


def risk_fraction_size(equity: float, risk_fraction: float, entry_price: float, stop_price: float) -> float:
    """Return units sized so a stop hit loses the configured share of equity."""
    if equity <= 0 or not 0 < risk_fraction <= 1 or entry_price <= 0:
        raise ValueError("equity, risk_fraction, and entry_price are invalid")
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        raise ValueError("stop price must differ from entry price")
    return equity * risk_fraction / risk_per_unit


def fractional_exposure(risk_fraction: float, stop_distance_fraction: float) -> float:
    """Convert fractional account risk and stop distance to capped notional exposure."""
    if not 0 < risk_fraction <= 1 or stop_distance_fraction <= 0:
        raise ValueError("risk_fraction and stop_distance_fraction are invalid")
    return min(1.0, risk_fraction / stop_distance_fraction)


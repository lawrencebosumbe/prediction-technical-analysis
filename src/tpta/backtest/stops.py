"""Stop and target price helpers."""

from __future__ import annotations


def stop_target(entry: float, atr: float, direction: int, stop_atr: float = 2.0, target_atr: float = 4.0) -> tuple[float, float]:
    """Calculate stop and target levels for a long or short position."""
    if entry <= 0 or atr <= 0 or direction not in {-1, 1} or min(stop_atr, target_atr) <= 0:
        raise ValueError("entry, ATR, direction, stop multiple, and target multiple must be valid")
    return entry - direction * stop_atr * atr, entry + direction * target_atr * atr


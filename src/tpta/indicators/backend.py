"""Optional indicator backend selection."""

from __future__ import annotations


def active() -> str:
    """Report the active backend; current indicator functions use pandas/NumPy."""
    return "numpy"


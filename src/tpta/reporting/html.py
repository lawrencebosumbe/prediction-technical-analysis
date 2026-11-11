"""Minimal self-contained HTML report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_html_report(metrics: dict[str, float | int], equity: pd.Series, output: str | Path) -> Path:
    """Write a small report containing summary metrics and an embedded equity chart."""
    from tpta.reporting.plots import plot_equity

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = output.with_suffix(".png")
    figure = plot_equity(equity, image)
    import matplotlib.pyplot as plt

    plt.close(figure)
    body = pd.DataFrame([metrics]).to_html(index=False, border=0, float_format=lambda value: f"{value:.4f}")
    output.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>TPTA report</title></head><body><h1>Backtest report</h1>{body}<img alt='Equity curve' src='{image.name}' style='max-width:100%'></body></html>", encoding="utf-8")
    return output


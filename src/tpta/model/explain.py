"""Portable model explanation helpers."""

from __future__ import annotations

import pandas as pd


def feature_importance(trainer: object) -> pd.Series:
    """Return coefficients or built-in feature importances when available."""
    model = getattr(trainer, "model", None)
    columns = getattr(trainer, "columns", None)
    if model is None or not columns:
        raise ValueError("a fitted trainer with feature names is required")
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "coef_"):
        values = abs(estimator.coef_).mean(axis=0)
    if values is None:
        raise ValueError("the fitted model does not expose feature importance")
    return pd.Series(values, index=columns, name="importance").sort_values(ascending=False)


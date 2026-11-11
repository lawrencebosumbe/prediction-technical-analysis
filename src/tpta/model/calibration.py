"""Probability calibration helpers."""

from __future__ import annotations

import pandas as pd


def calibrate_isotonic(probability: pd.Series, target: pd.Series) -> pd.Series:
    """Calibrate probabilities with isotonic regression."""
    from sklearn.isotonic import IsotonicRegression

    p, y = probability.align(target, join="inner")
    valid = p.notna() & y.notna()
    if valid.sum() < 10:
        raise ValueError("at least 10 labeled observations are required for calibration")
    fitted = IsotonicRegression(out_of_bounds="clip").fit(p.loc[valid], y.loc[valid].astype(int))
    return pd.Series(fitted.predict(probability.fillna(0.5)), index=probability.index, name="calibrated_probability")


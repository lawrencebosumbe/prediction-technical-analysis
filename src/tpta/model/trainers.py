"""Small optional machine-learning baselines."""

from __future__ import annotations

import pandas as pd


class LogisticRegressionTrainer:
    """Fit a deterministic scikit-learn logistic regression classifier."""

    def __init__(self, **params: object) -> None:
        self.params = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42, **params}
        self.model = None
        self.columns: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LogisticRegressionTrainer":
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError("Install the optional ML extra: pip install '.[ml]'") from exc
        data = features.loc[:, features.notna().any()]
        mask = target.notna() & data.notna().all(axis=1)
        if mask.sum() < 10 or target.loc[mask].nunique() < 2:
            raise ValueError("training data must contain at least 10 rows and two target classes")
        self.columns = list(data.columns)
        self.model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(**self.params))
        self.model.fit(data.loc[mask], target.loc[mask].astype(int))
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("fit must be called before predict_proba")
        if not self.columns:
            raise RuntimeError("model has no feature columns")
        probabilities = self.model.predict_proba(features.reindex(columns=self.columns))[:, 1]
        return pd.Series(probabilities, index=features.index, name="probability")


class LightGBMTrainer(LogisticRegressionTrainer):
    """Optional gradient-boosted tree classifier backed by LightGBM."""

    def __init__(self, features: str | None = None, target: str | None = None, **params: object) -> None:
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42, **params}
        self.model = None
        self.columns = []
        self.feature_set = features
        self.target_name = target

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LightGBMTrainer":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("Install LightGBM with: pip install '.[ml]'") from exc
        data = features.loc[:, features.notna().any()]
        mask = target.notna() & data.notna().all(axis=1)
        if mask.sum() < 10 or target.loc[mask].nunique() < 2:
            raise ValueError("training data must contain at least 10 rows and two target classes")
        self.columns = list(data.columns)
        self.model = LGBMClassifier(**self.params)
        self.model.fit(data.loc[mask], target.loc[mask].astype(int))
        return self


class XGBoostTrainer(LogisticRegressionTrainer):
    """Optional gradient-boosted tree classifier backed by XGBoost."""

    def __init__(self, **params: object) -> None:
        self.params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
            **params,
        }
        self.model = None
        self.columns = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "XGBoostTrainer":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install XGBoost with: pip install '.[ml]'") from exc
        data = features.loc[:, features.notna().any()]
        mask = target.notna() & data.notna().all(axis=1)
        if mask.sum() < 10 or target.loc[mask].nunique() < 2:
            raise ValueError("training data must contain at least 10 rows and two target classes")
        self.columns = list(data.columns)
        self.model = XGBClassifier(**self.params)
        self.model.fit(data.loc[mask], target.loc[mask].astype(int))
        return self


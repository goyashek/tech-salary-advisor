"""Preprocessing and pipeline assembly.

Numeric columns are scaled, categoricals one-hot encoded, and the binary
skill / missing-indicator columns pass through untouched. Salary is
right-skewed, so we fit on log1p(salary) and invert on predict via
TransformedTargetRegressor.
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="passthrough",
    )


def make_estimator(model, numeric_features, categorical_features):
    """Preprocessor + model, without target transform (used as a stacking base)."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def build_pipeline(model, numeric_features, categorical_features, log_target=True):
    """Full estimator. With log_target, fit on log1p(salary) and invert on predict."""
    estimator = make_estimator(model, numeric_features, categorical_features)
    if log_target:
        return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    return estimator

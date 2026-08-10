"""Preprocessing and pipeline assembly.

Numeric columns are scaled, categoricals one-hot encoded, and the binary
skill / missing-indicator columns pass through untouched. Salary is
right-skewed, so we fit on log1p(salary) and invert on predict via
TransformedTargetRegressor.
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features, categorical_features):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
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


def build_pipeline(model, numeric_features, categorical_features):
    """Fit on log1p(salary) and invert the transform during prediction."""
    estimator = make_estimator(model, numeric_features, categorical_features)
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)

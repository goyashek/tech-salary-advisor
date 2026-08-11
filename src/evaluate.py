"""Regression metrics used across training and reporting."""

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred, n_features=None):
    r2 = r2_score(y_true, y_pred)
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2,
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
    # Adjusted R2 discounts the score for the number of features used.
    if n_features:
        n = len(y_true)
        metrics["adj_r2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return metrics

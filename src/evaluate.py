"""Regression metrics used across training and reporting."""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def conformal_quantile(y_true, y_pred, level=0.9):
    """Return the finite-sample split-conformal residual quantile."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("calibration residuals must be finite and non-empty")
    rank = min(int(np.ceil((residuals.size + 1) * level)), residuals.size)
    return float(np.sort(residuals)[rank - 1])


def conformal_interval(predictions, half_width):
    """Build non-negative symmetric salary intervals in original INR units."""
    predictions = np.asarray(predictions, dtype=float)
    lower = np.maximum(0, predictions - float(half_width))
    return lower, np.maximum(lower, predictions + float(half_width))


def interval_metrics(y_true, lower, upper):
    y_true, lower, upper = map(np.asarray, (y_true, lower, upper))
    if not (len(y_true) == len(lower) == len(upper)) or len(y_true) == 0:
        raise ValueError("interval arrays must have the same non-zero length")
    return {
        "rows": int(len(y_true)),
        "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }


def interval_report(y_true, lower, upper, segments=None):
    """Report overall and optional segment-level interval performance."""
    y_true, lower, upper = map(np.asarray, (y_true, lower, upper))
    report = interval_metrics(y_true, lower, upper)
    if segments:
        report["segments"] = {}
        for name, labels in segments.items():
            labels = np.asarray(labels)
            if len(labels) != report["rows"]:
                raise ValueError("segment labels must match interval rows")
            report["segments"][name] = {
                str(label): interval_metrics(
                    y_true[labels == label],
                    lower[labels == label],
                    upper[labels == label],
                )
                for label in np.unique(labels)
            }
    return report


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

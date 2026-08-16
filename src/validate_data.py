"""Validate the raw salary data before deterministic cleanup."""

import hashlib
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import IMPUTED_COLUMNS

MAX_EXPERIENCE_YEARS = 100


class DataValidationError(ValueError):
    """Raised when the raw data violates the training data contract."""

    def __init__(self, result):
        self.result = result
        details = "; ".join(result["errors"])
        super().__init__(f"raw data validation failed: {details}")


def validate_data(df, target="Salary_INR"):
    """Return JSON-safe quality stats and fail on structural data errors."""
    required = [*IMPUTED_COLUMNS, target]
    missing_columns = [column for column in required if column not in df.columns]
    duplicate_rows = int(df.duplicated().sum())
    row_count = int(len(df))
    missingness = {column: int(value) for column, value in df.isna().sum().items()}
    result = {
        "status": "PASS",
        "required_columns": required,
        "row_count": row_count,
        "column_count": int(len(df.columns)),
        "missing_columns": missing_columns,
        "missingness": missingness,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rows / row_count if row_count else 0.0,
        "target": {
            "column": target,
            "missing_rows": int(df[target].isna().sum()) if target in df else None,
            "present_rows": int(df[target].notna().sum()) if target in df else None,
        },
        "numeric_ranges": {},
        "errors": [],
        "warnings": [],
    }

    if not row_count:
        result["errors"].append("dataset is empty")
    if missing_columns:
        result["errors"].append(f"missing required columns: {missing_columns}")

    rules = {
        "Experience_Years": (
            f"0 <= Experience_Years <= {MAX_EXPERIENCE_YEARS}",
            lambda values: (values < 0) | (values > MAX_EXPERIENCE_YEARS),
        ),
        "Salary_INR": ("Salary_INR > 0", lambda values: values <= 0),
    }
    for column, (rule, invalid_range) in rules.items():
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        non_numeric = df[column].notna() & values.isna()
        finite = values.notna() & np.isfinite(values)
        invalid = non_numeric | (values.notna() & ~np.isfinite(values))
        invalid |= finite & invalid_range(values)
        valid_values = values[finite]
        result["numeric_ranges"][column] = {
            "rule": rule,
            "observed_min": float(valid_values.min()) if len(valid_values) else None,
            "observed_max": float(valid_values.max()) if len(valid_values) else None,
            "invalid_rows": int(invalid.sum()),
        }
        if non_numeric.any():
            result["errors"].append(
                f"{column} contains {int(non_numeric.sum())} non-numeric values"
            )
        if (invalid & ~non_numeric).any():
            result["errors"].append(
                f"{column} violates its numeric range in "
                f"{int((invalid & ~non_numeric).sum())} rows"
            )

    for column, count in missingness.items():
        if count:
            result["warnings"].append(f"{column} has {count} missing values")
    if duplicate_rows:
        result["warnings"].append(f"dataset has {duplicate_rows} duplicate rows")

    if result["warnings"]:
        warnings.warn(
            "raw data quality warnings: " + "; ".join(result["warnings"]),
            UserWarning,
            stacklevel=2,
        )
    if result["errors"]:
        result["status"] = "FAIL"
        raise DataValidationError(result)
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def build_lineage(
    raw_path,
    config_path,
    validation,
    cleaned_rows,
    training_rows,
    test_rows,
    feature_count,
    validation_rows=0,
):
    """Build the small reproducibility record attached to the final run."""
    root = Path(__file__).resolve().parents[1]
    raw_path = Path(raw_path)
    config_path = Path(config_path)
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    if not config_path.is_absolute():
        config_path = root / config_path
    return {
        "git_sha": _git_sha(root),
        "raw_data_sha256": _sha256(raw_path),
        "config_sha256": _sha256(config_path),
        "python_version": platform.python_version(),
        "raw_rows": int(validation["row_count"]),
        "cleaned_rows": int(cleaned_rows),
        "training_rows": int(training_rows),
        "validation_rows": int(validation_rows),
        "test_rows": int(test_rows),
        "feature_count": int(feature_count),
    }

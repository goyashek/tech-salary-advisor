"""Checks for cleaning, validation, imputation, and outlier capping."""

import json
import numpy as np
import pandas as pd
import pytest

from src.data import (
    cap_outliers_iqr,
    clean,
    clean_education,
    clean_job_title,
    clean_location,
)
from src.pipeline import build_preprocessor
from src.validate_data import DataValidationError, build_lineage, validate_data


def test_title_mapping():
    assert clean_job_title(" DATA SCIENTIST ") == "Data Scientist"
    assert clean_job_title("ml engineer") == "Machine Learning Engineer"
    assert clean_job_title("something random") == "Software Engineer"  # fallback
    assert pd.isna(clean_job_title(None))


def test_location_and_education_mapping():
    assert clean_location(" noida ") == "Noida"
    assert clean_location("BANGALORE") == "Bangalore"
    assert clean_education("btech") == "Bachelor's"
    assert clean_education("m.tech") == "Master's"
    assert clean_education("PhD") == "PhD"


def test_clean_drops_missing_target_but_keeps_learned_fields_missing():
    raw = pd.DataFrame(
        {
            "Job_Title": ["data scientist", "qa", "backend"],
            "Experience_Years": [3.0, np.nan, 5.0],
            "Education_Level": ["btech", None, "phd"],
            "Location": ["noida", "mumbai", None],
            "Skills": ["Python, SQL", None, "Java"],
            "Salary_INR": [1_000_000, 900_000, None],  # last row dropped
        }
    )
    out = clean(raw)
    assert len(out) == 2  # row with missing salary is gone
    assert "Experience_Years_missing" in out.columns  # indicator flag added
    assert out.loc[1, "Experience_Years_missing"] == 1
    assert pd.isna(out.loc[1, "Experience_Years"])
    assert pd.isna(out.loc[1, "Education_Level"])
    assert out.loc[1, "Skills"] == ""


def test_preprocessor_learns_imputation_from_its_fit_data():
    train = pd.DataFrame(
        {
            "Experience_Years": [1.0, np.nan, 5.0],
            "Location": ["Pune", None, "Pune"],
        }
    )
    preprocessor = build_preprocessor(["Experience_Years"], ["Location"])
    preprocessor.fit(train)

    numeric_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    categorical_imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]
    assert numeric_imputer.statistics_[0] == 3.0
    assert categorical_imputer.statistics_[0] == "Pune"


def test_cap_outliers_clips_tail():
    y = pd.Series([1, 2, 3, 4, 5, 1000])
    capped = cap_outliers_iqr(y)
    assert capped.max() < 1000


def test_valid_data_passes_and_reports_quality_stats():
    raw = pd.DataFrame(
        {
            "Job_Title": ["data scientist", "qa"],
            "Experience_Years": [3.0, np.nan],
            "Education_Level": ["btech", "btech"],
            "Location": ["noida", "mumbai"],
            "Skills": ["Python", "SQL"],
            "Salary_INR": [1_000_000, 900_000],
        }
    )

    with pytest.warns(UserWarning, match="Experience_Years"):
        result = validate_data(raw)

    assert result["status"] == "PASS"
    assert result["row_count"] == 2
    assert result["missingness"]["Experience_Years"] == 1
    assert result["duplicate_rate"] == 0


def test_validation_fails_for_missing_columns():
    raw = pd.DataFrame({"Job_Title": ["data scientist"]})

    with pytest.raises(DataValidationError) as exc_info:
        validate_data(raw)

    assert "Salary_INR" in exc_info.value.result["missing_columns"]


def test_validation_fails_for_invalid_numeric_ranges():
    raw = pd.DataFrame(
        {
            "Job_Title": ["data scientist", "qa"],
            "Experience_Years": [-1, 101],
            "Education_Level": ["btech", "btech"],
            "Location": ["noida", "mumbai"],
            "Skills": ["Python", "SQL"],
            "Salary_INR": [1_000_000, -1],
        }
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_data(raw)

    assert exc_info.value.result["numeric_ranges"]["Salary_INR"]["invalid_rows"] == 1


def test_validation_and_lineage_are_json_safe(tmp_path):
    raw_path = tmp_path / "salary.csv"
    config_path = tmp_path / "config.yaml"
    raw_path.write_bytes(b"raw data")
    config_path.write_bytes(b"config")
    validation = validate_data(
        pd.DataFrame(
            {
                "Job_Title": ["qa"],
                "Experience_Years": [1],
                "Education_Level": ["btech"],
                "Location": ["pune"],
                "Skills": ["Python"],
                "Salary_INR": [500_000],
            }
        )
    )

    lineage = build_lineage(
        raw_path,
        config_path,
        validation,
        cleaned_rows=1,
        training_rows=1,
        test_rows=0,
        feature_count=3,
    )

    json.dumps({"validation": validation, "lineage": lineage})
    assert len(lineage["raw_data_sha256"]) == 64
    assert lineage["raw_rows"] == 1

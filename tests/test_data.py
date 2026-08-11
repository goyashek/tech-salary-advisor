"""Checks for the cleaning step: category mapping, imputation, outlier capping."""

import numpy as np
import pandas as pd

from src.data import (
    cap_outliers_iqr,
    clean,
    clean_education,
    clean_job_title,
    clean_location,
)
from src.pipeline import build_preprocessor


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

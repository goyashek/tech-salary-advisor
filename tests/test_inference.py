import numpy as np

from src.inference import (
    build_feature_row,
    predict_salary_inr,
    predict_salary_interval,
)


def metadata():
    skills = ["Python", "SQL", "Docker"]
    return {
        "feature_columns": [
            "Job_Title",
            "Experience_Years",
            "Education_Level",
            "Location",
            "Job_Title_missing",
            "Experience_Years_missing",
            "Education_Level_missing",
            "Location_missing",
            "Skills_missing",
            *skills,
            "skill_count",
        ],
        "all_skills": skills,
        "job_titles": ["Software Engineer", "Data Scientist"],
        "locations": ["Bangalore", "Noida"],
        "education_levels": ["Bachelor's", "Master's"],
    }


def test_build_feature_row_canonicalizes_and_orders_columns():
    row = build_feature_row(
        " data scientist ",
        3,
        "btech",
        " noida ",
        ["python", "SQL", "python"],
        metadata(),
    )

    assert list(row.columns) == metadata()["feature_columns"]
    assert row.loc[0, "Job_Title"] == "Data Scientist"
    assert row.loc[0, "Location"] == "Noida"
    assert row.loc[0, "Python"] == 1
    assert row.loc[0, "Docker"] == 0
    assert row.loc[0, "skill_count"] == 2
    assert row.loc[0, "Skills_missing"] == 0


def test_build_feature_row_rejects_unknown_skill():
    try:
        build_feature_row(
            "Data Scientist", 3, "Bachelor's", "Noida", ["Rust"], metadata()
        )
    except ValueError as exc:
        assert "unsupported skill" in str(exc)
    else:
        raise AssertionError("unknown skills should be rejected")


def test_exported_model_returns_numeric_prediction():
    prediction = predict_salary_inr(
        "Data Scientist", 3, "Bachelor's", "Bangalore", ["Python", "SQL"]
    )
    assert np.isfinite(prediction)
    assert prediction > 0


def test_prediction_interval_uses_exported_conformal_width():
    class ConstantModel:
        def predict(self, row):
            return np.array([1_000_000.0])

    assets = (
        ConstantModel(),
        {**metadata(), "prediction_interval": {"level": 0.9, "quantile_inr": 100_000}},
    )
    result = predict_salary_interval(
        "Data Scientist", 3, "Bachelor's", "Bangalore", ["Python"], assets
    )

    assert result == {
        "salary_inr": 1_000_000.0,
        "level": 0.9,
        "lower_inr": 900_000.0,
        "upper_inr": 1_100_000.0,
    }

"""Checks for skill parsing and feature construction."""
import pandas as pd

from src.features import add_skill_count, add_skill_flags, build_features

SKILLS = ["Python", "SQL", "Docker", "C++"]


def test_skill_flags_are_binary_and_substring_matched():
    df = pd.DataFrame({"Skills": ["Python, SQL", "docker", "C++, python"]})
    out = add_skill_flags(df, SKILLS)
    assert out.loc[0, "Python"] == 1 and out.loc[0, "Docker"] == 0
    assert out.loc[1, "Docker"] == 1
    assert out.loc[2, "C++"] == 1 and out.loc[2, "Python"] == 1  # case-insensitive


def test_skill_count_matches_listed_skills():
    df = pd.DataFrame({"Skills": ["Python, SQL, Docker", "Java", ""]})
    out = add_skill_count(df)
    assert list(out["skill_count"]) == [3, 1, 0]


def test_build_features_drops_raw_skills():
    df = pd.DataFrame({"Skills": ["Python, SQL"], "Salary_INR": [1_000_000]})
    out = build_features(df, SKILLS)
    assert "Skills" not in out.columns
    assert "skill_count" in out.columns and "Python" in out.columns

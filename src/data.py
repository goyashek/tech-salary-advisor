"""Load the raw CSV and clean it into a tidy frame.

The raw file has mixed casing, stray spaces, synonym job titles, and missing
values. This module handles deterministic cleanup; learned imputation stays in
the sklearn pipeline.
"""
import pandas as pd

# Record whether each input was reported before the pipeline imputes it.
IMPUTED_COLUMNS = ["Job_Title", "Experience_Years", "Education_Level", "Location", "Skills"]


def load_raw(path):
    return pd.read_csv(path)


def clean_job_title(title):
    if pd.isna(title):
        return title
    title = str(title).strip().lower()
    if "data scientist" in title:
        return "Data Scientist"
    if "ml engineer" in title or "machine learning" in title or "ai engineer" in title:
        return "Machine Learning Engineer"
    if "ai researcher" in title:
        return "AI Researcher"
    if "nlp" in title:
        return "NLP Engineer"
    if "cv engineer" in title or "computer vision" in title:
        return "Computer Vision Engineer"
    if "dl engineer" in title or "deep learning" in title:
        return "Deep Learning Engineer"
    if "frontend" in title:
        return "Frontend Developer"
    if "backend" in title:
        return "Backend Developer"
    if "fullstack" in title or "full stack" in title:
        return "Full Stack Developer"
    if "devops" in title:
        return "DevOps Engineer"
    if "data engineer" in title:
        return "Data Engineer"
    if "qa" in title:
        return "QA Engineer"
    if "product" in title or "pm" in title:
        return "Product Manager"
    return "Software Engineer"


def clean_location(city):
    if pd.isna(city):
        return city
    city = str(city).strip().lower()
    for key, label in [
        ("bangalore", "Bangalore"),
        ("mumbai", "Mumbai"),
        ("delhi", "Delhi NCR"),
        ("noida", "Noida"),
        ("hyderabad", "Hyderabad"),
        ("pune", "Pune"),
        ("chennai", "Chennai"),
    ]:
        if key in city:
            return label
    return "Bangalore"


def clean_education(edu):
    if pd.isna(edu):
        return edu
    edu = str(edu).strip().lower()
    if any(k in edu for k in ["master", "mtech", "m.tech", "ms", "m.s."]):
        return "Master's"
    if "phd" in edu or "doctor" in edu:
        return "PhD"
    return "Bachelor's"


def clean(df, target="Salary_INR"):
    """Drop missing targets, add flags, and standardize deterministic values."""
    df = df.copy()
    df = df.dropna(subset=[target])

    for col in IMPUTED_COLUMNS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # An empty skill list is a fixed representation, not a learned statistic.
    df["Skills"] = df["Skills"].fillna("")

    df["Job_Title"] = df["Job_Title"].apply(clean_job_title)
    df["Location"] = df["Location"].apply(clean_location)
    df["Education_Level"] = df["Education_Level"].apply(clean_education)
    return df


def cap_outliers_iqr(y, factor=1.5):
    """Clip a series to the IQR fence. Use on the training target only."""
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    return y.clip(q1 - factor * iqr, q3 + factor * iqr)

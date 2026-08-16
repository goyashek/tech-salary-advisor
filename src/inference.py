"""Load the exported model and turn a career profile into one prediction row."""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data import clean_education, clean_job_title, clean_location
from src.model_registry import configure_mlflow


def _load_registry_assets(cfg):
    import mlflow
    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient

    tracking_uri = configure_mlflow(cfg, mlflow)
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri[len("sqlite:///") :])
        if not db_path.exists():
            raise FileNotFoundError(db_path)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    model_name = cfg["mlflow"]["registered_model"]
    alias = cfg["mlflow"]["champion_alias"]
    version = client.get_model_version_by_alias(model_name, alias)
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    metadata_path = download_artifacts(
        run_id=version.run_id,
        artifact_path="model/metadata.json",
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )
    with Path(metadata_path).open() as handle:
        metadata = json.load(handle)
    return model, metadata


@lru_cache(maxsize=1)
def load_model_assets(model_path=None, metadata_path=None):
    """Load model assets once per process."""
    cfg = load_config()
    if model_path is None and metadata_path is None:
        try:
            return _load_registry_assets(cfg)
        # ponytail: broad fallback keeps local serving alive; narrow this to
        # MLflow errors if registry diagnostics become important.
        except Exception:
            pass
    model_path = (
        ROOT / cfg["output"]["model"] if model_path is None else Path(model_path)
    )
    metadata_path = (
        ROOT / cfg["output"]["metadata"]
        if metadata_path is None
        else Path(metadata_path)
    )
    return joblib.load(model_path), joblib.load(metadata_path)


def _category(value, cleaner, allowed, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    value = cleaner(value)
    if value not in allowed:
        raise ValueError(f"unsupported {field}: {value}")
    return value


def _skills(values, allowed):
    if values is None or isinstance(values, str):
        raise ValueError("skills must be a list of known skill names")
    lookup = {skill.casefold(): skill for skill in allowed}
    selected = []
    for value in values:
        if not isinstance(value, str) or value.strip().casefold() not in lookup:
            raise ValueError(f"unsupported skill: {value}")
        skill = lookup[value.strip().casefold()]
        if skill not in selected:
            selected.append(skill)
    return selected


def build_feature_row(job_title, experience, education, location, skills, metadata):
    """Build the exact feature columns expected by the exported pipeline."""
    try:
        experience = float(experience)
    except (TypeError, ValueError) as exc:
        raise ValueError("experience must be a finite, non-negative number") from exc
    if not np.isfinite(experience) or experience < 0:
        raise ValueError("experience must be a finite, non-negative number")

    job_title = _category(
        job_title, clean_job_title, metadata["job_titles"], "job title"
    )
    education = _category(
        education, clean_education, metadata["education_levels"], "education"
    )
    location = _category(location, clean_location, metadata["locations"], "location")
    skills = _skills(skills, metadata["all_skills"])

    values = {
        "Job_Title": job_title,
        "Experience_Years": experience,
        "Education_Level": education,
        "Location": location,
        "Job_Title_missing": 0,
        "Experience_Years_missing": 0,
        "Education_Level_missing": 0,
        "Location_missing": 0,
        "Skills_missing": 0,
        "skill_count": len(skills),
    }
    values.update({skill: int(skill in skills) for skill in metadata["all_skills"]})
    return pd.DataFrame([values]).reindex(columns=metadata["feature_columns"])


def predict_salary_inr(job_title, experience, education, location, skills, assets=None):
    """Return the raw salary estimate in INR for one career profile."""
    model, metadata = assets or load_model_assets()
    row = build_feature_row(
        job_title, experience, education, location, skills, metadata
    )
    return float(model.predict(row)[0])

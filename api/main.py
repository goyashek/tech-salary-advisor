"""FastAPI entry point for salary predictions."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import predict_salary_interval

app = FastAPI(title="Tech Salary Advisor API")


class SalaryProfile(BaseModel):
    job_title: str
    experience_years: float = Field(ge=0, allow_inf_nan=False)
    education: str
    location: str
    skills: list[str] = Field(default_factory=list)


class SalaryPrediction(BaseModel):
    salary_inr: int
    salary_lpa: float
    interval_level: float
    interval_lower_inr: int
    interval_upper_inr: int
    interval_lower_lpa: float
    interval_upper_lpa: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=SalaryPrediction)
def predict(profile: SalaryProfile) -> SalaryPrediction:
    try:
        result = predict_salary_interval(
            profile.job_title,
            profile.experience_years,
            profile.education,
            profile.location,
            profile.skills,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SalaryPrediction(
        salary_inr=round(result["salary_inr"]),
        salary_lpa=round(result["salary_inr"] / 100_000, 2),
        interval_level=result["level"],
        interval_lower_inr=round(result["lower_inr"]),
        interval_upper_inr=round(result["upper_inr"]),
        interval_lower_lpa=round(result["lower_inr"] / 100_000, 2),
        interval_upper_lpa=round(result["upper_inr"] / 100_000, 2),
    )

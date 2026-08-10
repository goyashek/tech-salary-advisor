"""FastAPI entry point for salary predictions."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import predict_salary_inr

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=SalaryPrediction)
def predict(profile: SalaryProfile) -> SalaryPrediction:
    try:
        estimate = predict_salary_inr(
            profile.job_title,
            profile.experience_years,
            profile.education,
            profile.location,
            profile.skills,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SalaryPrediction(
        salary_inr=round(estimate),
        salary_lpa=round(estimate / 100_000, 2),
    )

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_returns_inr_and_lpa():
    response = client.post(
        "/predict",
        json={
            "job_title": "Data Scientist",
            "experience_years": 3,
            "education": "Bachelor's",
            "location": "Bangalore",
            "skills": ["Python", "SQL"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["salary_inr"] > 0
    assert body["salary_lpa"] == round(body["salary_inr"] / 100_000, 2)


def test_prediction_rejects_malformed_request():
    response = client.post("/predict", json={"job_title": "Data Scientist"})

    assert response.status_code == 422


def test_prediction_rejects_unknown_skill():
    response = client.post(
        "/predict",
        json={
            "job_title": "Data Scientist",
            "experience_years": 3,
            "education": "Bachelor's",
            "location": "Bangalore",
            "skills": ["Rust"],
        },
    )

    assert response.status_code == 422

<div align="center">

# 💼 Tech Salary Advisor

### Explainable salary estimation with a reproducible ML pipeline

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-337AB7)
![CatBoost](https://img.shields.io/badge/CatBoost-tuned-FFCC00)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://tech-salary-advisor.streamlit.app/)
[![CI](https://github.com/goyashek/Tech-Salary-Advisor/actions/workflows/ci.yml/badge.svg)](https://github.com/goyashek/Tech-Salary-Advisor/actions/workflows/ci.yml)

![Best Model](https://img.shields.io/badge/Best%20Model-Stacking%20(XGB%2BCatBoost)-success)
![R2](https://img.shields.io/badge/Held--out%20R²-0.885-blue)
![MAE](https://img.shields.io/badge/MAE-%E2%82%B91.55%20LPA-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## At a glance

| Area | What this project demonstrates |
|---|---|
| Modeling | Regression, log-target transformation, Optuna, and stacking |
| Evaluation | Training-only cross-validation and untouched held-out testing |
| Explainability | Permutation importance and counterfactual profile comparisons |
| Serving | Streamlit UI and typed FastAPI API |
| MLOps | Raw-data validation, MLflow lineage, Docker runtime, CI, tests, and pinned API dependencies |

- **Task:** Estimate annual salary in INR from role, experience, education, location, and skills.
- **Dataset:** Approximately 110,000 public salary records; 107,735 remained after removing rows without a target.
- **Best model:** XGBoost + CatBoost stacking ensemble selected by training-only cross-validation.
- **Held-out R²:** 0.8849
- **Held-out MAE:** ₹155,014

## 🎥 Demo

https://github.com/user-attachments/assets/6d0e09a7-e7c3-4c38-8589-7d68b8e18c0d

## 📌 Overview

This is an end-to-end salary estimation system built around a public Indian technology salary dataset. The final system includes both the modeling work and the engineering required to make the model reproducible, inspectable, and accessible through multiple interfaces.

I built it as a learning progression: understand the data first, make the evaluation trustworthy, compare models deliberately, inspect what the model relies on, and then connect the exported artifact to a real application and API.

The model produces estimates from patterns in the project dataset. It is not an authoritative market-rate system and should not be treated as a compensation benchmark.

## 🧭 Project journey

The project did not begin as a polished application. It started with a dirty salary file and a simple modeling goal.

The early work focused on understanding the dataset: which columns were incomplete, how salary was distributed, how inconsistent labels should be standardized, and whether skills added useful signal after role and experience were known.

Once the baseline pipeline worked, I added a raw-data validation gate so schema and numeric range problems fail before cleanup can hide them. Missingness and duplicate rates are recorded as quality statistics. I then moved toward model selection: instead of choosing a model from the test set, I used cross-validation on the training data, tuned the strongest boosting models with Optuna, and evaluated the selected stack once on the untouched test split.

The final engineering work connected the trained artifact to Streamlit and FastAPI through one shared inference path, then added Docker, tests, linting, and GitHub Actions.

The commit history reflects that progression:

```text
dataset and scaffold
        ↓
EDA, cleaning, and feature engineering
        ↓
preprocessing pipeline and baseline models
        ↓
Optuna tuning, stacking, and feature importance
        ↓
model export and Streamlit app
        ↓
tests, documentation, and methodology cleanup
        ↓
shared inference, FastAPI, Docker, and CI
```

## 📊 Data and task framing

The model predicts `Salary_INR`, an annual salary value in Indian rupees. The input fields are:

- Job title
- Years of experience
- Education level
- Location
- Technical skills

| Property | Details |
|---|---|
| Source | [Indian Tech Salaries](https://www.kaggle.com/datasets/ashishprajapati223/indian-tech-salaries/data) on Kaggle |
| Raw size | Approximately 110,000 rows |
| Usable rows | 107,735 after dropping missing salary targets |
| Target | `Salary_INR` |
| Features | `Job_Title`, `Experience_Years`, `Education_Level`, `Location`, and `Skills` |

This is a supervised regression problem, not a salary-market benchmark. The dataset provides examples of salary patterns across roles, cities, education levels, experience, and listed skills. It does not contain every factor that affects compensation, such as company, industry, negotiation, equity, employment gaps, or current market conditions.

## 🧹 Why these preprocessing decisions?

- **Drop missing targets:** supervised learning cannot use rows without a known salary.
- **Standardize text deterministically:** casing, spacing, and known synonyms should not create artificial categories.
- **Keep missing indicators:** a missing field may itself carry information.
- **Impute inside the pipeline:** medians and most-frequent categories are learned separately inside each training fold.
- **Cap only training targets:** extreme salary values are clipped using a training-derived IQR fence; the test target remains untouched.
- **Expand skills into flags:** the comma-separated skill field becomes binary skill columns plus `skill_count`.

The important distinction was between transformations that can be applied safely before splitting and statistics that must be learned after splitting. Whitespace cleanup, casing normalization, and known synonym mapping are deterministic. Imputation is learned from data, so it belongs inside the sklearn pipeline.

## 🧠 Model progression and advanced techniques

The model progression was deliberately incremental:

1. **ElasticNet** as a simple linear baseline.
2. **Random Forest** as a nonlinear tree ensemble.
3. **XGBoost and CatBoost** as stronger boosting candidates.
4. **Optuna tuning** for XGBoost and CatBoost using a seeded TPE search.
5. **RidgeCV stacking** to combine the two tuned learners.

Several techniques were used because they addressed specific properties of the problem:

- `log1p` target transformation for the right-skewed salary distribution.
- `TransformedTargetRegressor` to keep the target transformation attached to the estimator.
- Median and most-frequent imputation inside sklearn pipelines.
- One-hot encoding for categorical profile fields.
- CatBoost and XGBoost for nonlinear interactions between career variables.
- Training-only IQR target capping to reduce the influence of extreme salary values.
- Seeded Optuna tuning and fixed train/test splits for reproducibility.
- RidgeCV as the stacking meta-model.

The final modeling path is:

```text
profile features
      ↓
train-fitted preprocessing
      ↓
tuned XGBoost ─┐
                ├── RidgeCV stacking model
tuned CatBoost ┘
      ↓
salary estimate in INR
```

## 🔍 Explainability and xAI

The project uses lightweight, model-agnostic explainability rather than treating the prediction as a black box.

Permutation importance on the held-out split shows that experience and job title carry the strongest signal, followed by education and location. Individual skill flags and `skill_count` contribute much less once the main career variables are already known.

The Streamlit app also provides a counterfactual skill comparison: it adds one available skill at a time and reports how much the model estimate changes. This is a model-estimated difference, not a causal promise that learning the skill will produce that salary increase.

These explanations describe model behavior. They should not be interpreted as causal salary effects.

## 🏆 Results

Models were compared with three-fold cross-validation on a fixed 30,000-row subset of the training split.

| Model | Mean CV R² |
|---|---:|
| ElasticNet | 0.7745 |
| Random Forest | 0.8248 |
| XGBoost | 0.8771 |
| CatBoost | 0.8829 |
| Tuned XGBoost | 0.8808 |
| Tuned CatBoost | 0.8838 |
| **Stacking** | **0.8838** |

The stacking model was selected using the training split and evaluated once on the untouched 21,547-row test split:

- **R²:** 0.8849
- **MAE:** ₹155,014
- **RMSE:** ₹207,766
- **MAPE:** 10.31%

The score is the result of one seeded training run on one public dataset and one held-out split. It should not be read as a universal salary-accuracy guarantee.

## 🗂️ Notebook progression

| Notebook | Purpose |
|---|---|
| [`EDA.ipynb`](notebooks/EDA.ipynb) | Investigates missing values, salary skew, outliers, inconsistent labels, feature relationships, and skill frequencies. |
| [`Salary_Prediction.ipynb`](notebooks/Salary_Prediction.ipynb) | Rebuilds the shared cleaning and feature pipeline, compares baseline models, and loads the exported final metadata. |

The notebooks show the reasoning, while `src/` contains the reusable implementation used by training and serving.

The notebook workflow follows the same order as the project itself:

```text
inspect the data
      ↓
make deterministic cleanup decisions
      ↓
split before learned preprocessing
      ↓
compare models with CV
      ↓
load the exported result
```

## 🧩 MLOps and serving

The MLOps layer is intentionally small but complete for this project:

```text
training code
    ↓
raw CSV validation
    ↓
salary_model.pkl + metadata.pkl
    ↓
shared inference module
    ├── Streamlit UI
    └── FastAPI API
              ↓
          Docker runtime
```

The goal was not to add infrastructure for its own sake. Each layer solves a practical problem in the model’s path from experiment to use:

- MLflow tracks model-comparison runs, CV metrics, final metrics, selected parameters where applicable, and exported artifacts.
- Raw-data validation checks required columns, numeric ranges, missingness, and duplicate rates before cleanup. Structural errors stop training; quality warnings are retained in the final run.
- The final MLflow run stores machine-readable `data_validation.json` and `lineage.json` artifacts, including Git SHA, raw-data SHA-256, config hash, Python version, row counts, and feature count.
- `metadata.pkl` stores feature order, category options, skill names, metrics, split sizes, validation results, and lineage.
- `src/inference.py` is the single path for validation, feature-row construction, column ordering, and prediction.
- Streamlit uses the shared inference module for the interactive app.
- FastAPI exposes `/health` and `/predict`.
- Docker packages the API with pinned runtime dependencies.
- Ruff, pre-commit, pytest, and GitHub Actions automate quality checks.

## 🚀 Quickstart

### Run the complete local path

```bash
pip install -r requirements-dev.txt
pytest -q
python -m src.train
streamlit run streamlit/app.py
docker build -t tech-salary-advisor .
docker run --rm -p 8000:8000 tech-salary-advisor
```

The full training run performs model comparison, Optuna tuning, stacking, evaluation, MLflow logging, and artifact export. For a shorter smoke run, use:

```bash
python -m src.train --fast
```

The exported model already ships in `streamlit/models/`, so retraining is optional for trying the app.

### Run the FastAPI service

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

Check the service and request a prediction:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"job_title\":\"Data Scientist\",\"experience_years\":3,\"education\":\"Bachelor's\",\"location\":\"Bangalore\",\"skills\":[\"Python\",\"SQL\"]}"
```

The response contains `salary_inr` and `salary_lpa`. The demo API is unauthenticated; do not expose it to sensitive data without adding authentication.

## 📁 Repository anatomy

The repository is split by responsibility:

- `notebooks/` explains the investigation.
- `src/` contains reusable ML logic.
- `tests/` protects the important data and inference paths.
- `streamlit/` contains the user-facing application and exported model artifacts.
- `api/` exposes the model over HTTP.
- `Dockerfile` and `.github/` cover deployment and automation.

```text
Tech-Salary-Advisor/
├── config.yaml                    # data, features, split, tuning, and output settings
├── data/
│   └── salary_dataset_dirty.csv   # raw public dataset
├── src/
│   ├── data.py                    # deterministic cleanup and missing indicators
│   ├── features.py                # skill flags and skill_count
│   ├── pipeline.py                # preprocessing and target transformation
│   ├── evaluate.py                # regression metrics
│   ├── validate_data.py            # raw-data gate and training lineage
│   ├── inference.py               # shared validation and prediction path
│   └── train.py                   # baselines, tuning, stacking, MLflow, and export
├── api/main.py                    # FastAPI service
├── streamlit/app.py               # interactive application
├── streamlit/models/              # exported model and metadata
├── notebooks/                     # EDA and modeling walkthroughs
├── tests/                         # data, model, inference, and API tests
├── Dockerfile                     # API runtime image
├── .github/workflows/ci.yml       # Ruff, pytest, and Docker CI
├── .pre-commit-config.yaml        # local quality hooks
├── requirements-api.txt           # pinned API runtime dependencies
├── requirements-dev.txt           # development and test dependencies
└── README.md
```

## 🛠️ Tools and acknowledgements

The project uses standard tools at each stage:

- pandas and NumPy for data work;
- scikit-learn for preprocessing, evaluation, and model composition;
- XGBoost and CatBoost for nonlinear regression;
- Optuna for tuning;
- MLflow for experiment tracking;
- Streamlit and FastAPI for serving;
- Docker and GitHub Actions for reproducibility and automation.

This project follows a learning-first approach: use established libraries where they fit, keep reusable logic in `src/`, and add infrastructure only when it improves reproducibility or serving. The modeling techniques follow concepts from CampusX’s *100 Days of Machine Learning* curriculum.

## ⚖️ Limitations

This is an educational salary estimation project, not an authoritative market-rate system. The data comes from one public dataset and one seeded training run. The model does not account for company, industry, seniority band, negotiation, equity, employment gaps, or changing market conditions.

The skill-comparison view shows model behavior, not causal salary effects. The API is unauthenticated and should not be exposed to sensitive data without an authentication layer.

Predictions should not be used as the sole basis for compensation, hiring, or career decisions.

## License

The original source code is released under the [MIT License](LICENSE). The dataset remains subject to its own terms on Kaggle.

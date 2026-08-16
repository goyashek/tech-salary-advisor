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

![Best Model](https://img.shields.io/badge/Best%20Model-Tuned%20CatBoost-success)
![R2](https://img.shields.io/badge/Held--out%20R²-0.884-blue)
![MAE](https://img.shields.io/badge/MAE-%E2%82%B91.55%20LPA-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## At a glance

| Area | What this project demonstrates |
|---|---|
| Modeling | Regression, log-target transformation, Optuna, sklearn/Keras ANN benchmarks, and complexity-aware selection |
| Evaluation | Training-only CV, promotion validation, and untouched final testing |
| Explainability | Permutation importance and counterfactual profile comparisons |
| Serving | Streamlit UI and typed FastAPI API |
| MLOps | Raw-data validation, MLflow registry/promotion, Docker runtime, CI, tests, and pinned API dependencies |

- **Task:** Estimate annual salary in INR from role, experience, education, location, and skills.
- **Dataset:** Approximately 110,000 public salary records; 107,735 remained after removing rows without a target.
- **Best model:** Tuned CatBoost; stacking did not earn its added complexity.
- **Held-out R²:** 0.8841
- **Held-out MAE:** ₹155,414

## 🎥 Demo

https://github.com/user-attachments/assets/6d0e09a7-e7c3-4c38-8589-7d68b8e18c0d

## 📌 Overview

This is an end-to-end salary estimation system built around a public Indian technology salary dataset. The final system includes both the modeling work and the engineering required to make the model reproducible, inspectable, and accessible through multiple interfaces.

I built it as a learning progression: understand the data first, make the evaluation trustworthy, compare models deliberately, inspect what the model relies on, and then connect the exported artifact to a real application and API.

The model produces estimates from patterns in the project dataset. It is not an authoritative market-rate system and should not be treated as a compensation benchmark.

## 🧭 Project journey

The project did not begin as a polished application. It started with a dirty salary file and a simple modeling goal.

The early work focused on understanding the dataset: which columns were incomplete, how salary was distributed, how inconsistent labels should be standardized, and whether skills added useful signal after role and experience were known.

Once the baseline pipeline worked, I added a raw-data validation gate so schema and numeric range problems fail before cleanup can hide them. Missingness and duplicate rates are recorded as quality statistics. I then moved toward model selection: instead of choosing a model from the test set, I used cross-validation on the training data, benchmarked a small neural network, tuned the strongest boosting models with Optuna, and evaluated the selected model once on the untouched test split.

The final engineering work connected the trained artifact to Streamlit and FastAPI through one shared inference path, then added validation, MLflow registry promotion, Docker, tests, linting, and GitHub Actions.

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
5. **A small sklearn MLP** as a neural-network performance/explainability benchmark.
6. **A standalone Keras ANN experiment** to test whether a stronger neural training recipe closes the gap.
7. **RidgeCV stacking** as a candidate that must beat the best single model by at least 0.002 CV R².

Several techniques were used because they addressed specific properties of the problem:

- `log1p` target transformation for the right-skewed salary distribution.
- `TransformedTargetRegressor` to keep the target transformation attached to the estimator.
- Median and most-frequent imputation inside sklearn pipelines.
- One-hot encoding for categorical profile fields.
- CatBoost and XGBoost for nonlinear interactions between career variables.
- Training-only IQR target capping to reduce the influence of extreme salary values.
- Seeded Optuna tuning and fixed train/test splits for reproducibility.
- Early stopping for the sklearn MLP benchmark.
- Training-fold target standardization, He initialization, BatchNorm, Dropout, AdamW, learning-rate reduction, and restored-best-weight early stopping for the Keras ANN.
- RidgeCV as a stacking candidate, with a material-gain rule before it can be selected.

The final modeling path is:

```text
profile features
      ↓
train-fitted preprocessing
      ↓
tuned CatBoost
      ↓
salary estimate in INR
```

## 🔍 Explainability and xAI

The project uses lightweight, model-agnostic explainability rather than treating the prediction as a black box.

Permutation importance on the held-out split shows that experience and job title carry the strongest signal, followed by education and location. Individual skill flags and `skill_count` contribute much less once the main career variables are already known.

The registered run's sklearn MLP benchmark reached 0.8638 CV R². More patience improved it to 0.8666, and a standalone Keras ANN experiment reached 0.8753 on the local Metal GPU. The stronger neural recipe closed most of the gap but still trailed tuned CatBoost's 0.8802 by 0.0049. It was only an experiment: it did not use promotion validation, enter the registry, affect serving, or touch the final test. Stacking reached 0.8803, only 0.0001 above tuned CatBoost, so the simpler single model won under the 0.002 minimum-gain rule.

The ANN also has the weakest xAI story in this comparison. Dense-layer weights do not map cleanly back to salary drivers after one-hot encoding and nonlinear interactions. Model-agnostic permutation importance could still be applied, but it would be post-hoc, can divide credit across correlated inputs, and would explain associations learned by the model rather than causal salary effects. That added explanation burden was not justified without a performance gain over CatBoost.

The Streamlit app also provides a counterfactual skill comparison: it adds one available skill at a time and reports how much the model estimate changes. This is a model-estimated difference, not a causal promise that learning the skill will produce that salary increase.

These explanations describe model behavior. They should not be interpreted as causal salary effects.

## 🏆 Results

Models were compared with three-fold cross-validation on a fixed 30,000-row subset of the training split.

| Model | Explainability | Mean CV R² |
|---|---|---:|
| ElasticNet | High | 0.7677 |
| Random Forest | Medium | 0.8185 |
| XGBoost | Medium | 0.8730 |
| CatBoost | Medium | 0.8790 |
| sklearn MLP | Low | 0.8638 |
| Keras ANN | Low | 0.8753 |
| Tuned XGBoost | Medium | 0.8775 |
| **Tuned CatBoost** | **Medium** | **0.8802** |
| Stacking | Low | 0.8803 |

Stacking's raw gain over tuned CatBoost was only 0.00006 CV R², below the configured 0.002 requirement. Tuned CatBoost was promoted with validation R² 0.8812 and validation MAE ₹158,032, then evaluated once on the untouched 21,547-row final test split:

- **R²:** 0.8841
- **MAE:** ₹155,414
- **RMSE:** ₹208,545
- **MAPE:** 10.32%

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
raw CSV validation
    ↓
training and CV
    ↓
MLflow candidate
    ↓
validation-only promotion gate
    ├── rejected → existing champion remains
    └── promoted → @champion
    ↓
shared inference module
    ├── Streamlit UI
    └── FastAPI API
              ↓
          Docker runtime
```

The goal was not to add infrastructure for its own sake. Each layer solves a practical problem in the model’s path from experiment to use:

- MLflow uses a local SQLite backend for model-comparison runs, candidate models, validation metrics, aliases, and artifacts.
- Raw-data validation checks required columns, numeric ranges, missingness, and duplicate rates before cleanup. Structural errors stop training; quality warnings are retained in the final run.
- The candidate is compared with `@champion` on a separate validation split: MAE may worsen by at most 2%, and R² may fall by at most 0.005. The final test split is reporting-only.
- A promoted candidate receives `@challenger` and `@champion`; a rejected candidate leaves the existing champion and local fallback untouched.
- Inference loads `models:/tech-salary-advisor@champion` when MLflow and the registry are available, then falls back to `streamlit/models/` for the Docker/local artifact path.
- The final MLflow run stores machine-readable `data_validation.json`, `lineage.json`, `promotion.json`, and model metadata artifacts.
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

To inspect the registry in another terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The full training run performs model comparison, Optuna tuning, stacking, validation-gated registry promotion, final evaluation, and local fallback export. For a shorter smoke run, use:

```bash
python -m src.train --fast
```

The exported model already ships in `streamlit/models/`, so retraining is optional for trying the app.

### Run the optional Keras ANN benchmark on Apple Silicon

```bash
conda create --prefix ./.venv python=3.12 pip -y
.venv/bin/pip install -r requirements-keras-macos.txt
make keras-benchmark
```

TensorFlow uses Metal automatically when the GPU is available. The recorded
three-fold run used the same 30,000-row training-only subset and did not inspect
promotion-validation or final-test labels.

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
│   ├── model_registry.py           # MLflow aliases and promotion gate
│   ├── validate_data.py            # raw-data gate and training lineage
│   ├── inference.py               # shared validation and prediction path
│   ├── keras_benchmark.py          # optional regularized Keras ANN comparison
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
├── requirements-keras-macos.txt   # optional Apple Silicon Keras environment
└── README.md
```

## 🛠️ Tools and acknowledgements

The project uses standard tools at each stage:

- pandas and NumPy for data work;
- scikit-learn for preprocessing, evaluation, and model composition;
- XGBoost and CatBoost for nonlinear regression;
- Keras/TensorFlow for the optional regularized ANN benchmark;
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

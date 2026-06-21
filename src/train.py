"""Train and compare baseline models, log to MLflow, export the best.

Run: python -m src.train
"""
import os
import warnings
from pathlib import Path

import joblib

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import ROOT, load_config
from src.data import cap_outliers_iqr, clean, load_raw
from src.evaluate import regression_metrics
from src.features import build_features
from src.pipeline import build_pipeline


def main(config_path="config.yaml"):
    import mlflow

    cfg = load_config(config_path)
    num, cat = cfg["numeric_features"], cfg["categorical_features"]

    df = build_features(clean(load_raw(ROOT / cfg["data"]["path"])), cfg["skills"])
    y = df[cfg["data"]["target"]]
    X = df.drop(columns=[cfg["data"]["target"]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=cfg["split"]["random_state"]
    )
    y_train = cap_outliers_iqr(y_train, cfg["outlier_iqr_factor"])

    mlflow.set_tracking_uri(f"file:{ROOT / cfg['mlflow']['tracking_dir']}")
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    models = {
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=cfg["random_forest"]["n_estimators"],
            max_depth=cfg["random_forest"]["max_depth"], random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
        "catboost": CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, random_seed=42, verbose=False, allow_writing_files=False),
    }

    best = None
    for name, model in models.items():
        pipe = build_pipeline(model, num, cat)
        pipe.fit(X_train, y_train)
        m = regression_metrics(y_test, pipe.predict(X_test))
        with mlflow.start_run(run_name=name):
            mlflow.log_metrics(m)
        print(f"{name:16} R2={m['r2']:.4f}  MAE={m['mae']:,.0f}")
        if best is None or m["r2"] > best[2]["r2"]:
            best = (name, pipe, m)

    name, pipe, m = best
    print("best:", name)
    metadata = {
        "feature_columns": list(X.columns),
        "all_skills": cfg["skills"],
        "median_exp": float(df["Experience_Years"].median()),
        "model_name": name,
        "mae": float(m["mae"]), "rmse": float(m["rmse"]),
        "r2": float(m["r2"]), "mape": float(m["mape"]),
        "job_titles": cfg["job_titles"], "locations": cfg["locations"],
        "education_levels": cfg["education_levels"],
    }
    for key in ("model", "metadata"):
        Path(ROOT / cfg["output"][key]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ROOT / cfg["output"]["model"])
    joblib.dump(metadata, ROOT / cfg["output"]["metadata"])


if __name__ == "__main__":
    main()

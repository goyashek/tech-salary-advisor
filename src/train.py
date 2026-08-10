"""Train, tune, compare, and export the salary model.

Run: python -m src.train  (add --fast for a quick smoke run)

Steps: deterministic clean -> split -> train-fitted preprocessing -> model
comparison by cross-validation -> one final test evaluation -> export. Model
comparison and final artifacts are logged to a local MLflow store.
"""
import argparse
import os
import warnings
from pathlib import Path

# Keep the simple local file store (./mlruns); no tracking server needed.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

from src.config import ROOT, load_config
from src.data import cap_outliers_iqr, clean, load_raw
from src.evaluate import regression_metrics
from src.features import build_features
from src.pipeline import build_pipeline, make_estimator

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _split(cfg):
    df = clean(load_raw(ROOT / cfg["data"]["path"]), target=cfg["data"]["target"])
    df = build_features(df, cfg["skills"])
    y = df[cfg["data"]["target"]]
    X = df.drop(columns=[cfg["data"]["target"]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=cfg["split"]["random_state"]
    )
    y_train = cap_outliers_iqr(y_train, cfg["outlier_iqr_factor"])
    median_exp = float(X_train["Experience_Years"].median())
    return X, X_train, X_test, y_train, y_test, median_exp


def _tune(build_model, space, X, y, num, cat, n_trials, cv):
    """Optuna Bayesian search over `space`, scoring cross-validated R2."""
    def objective(trial):
        model = build_model({k: fn(trial) for k, fn in space.items()})
        pipe = build_pipeline(model, num, cat)
        return cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=1).mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def select_best(results):
    """Return the candidate with the strongest cross-validation score."""
    return max(results, key=lambda name: results[name]["cv_r2"])


def main(config_path="config.yaml", fast=False):
    import mlflow

    cfg = load_config(config_path)
    num, cat = cfg["numeric_features"], cfg["categorical_features"]
    n_trials = 2 if fast else cfg["tuning"]["n_trials"]
    subset = 3000 if fast else cfg["tuning"]["subset_size"]
    cv = cfg["tuning"]["cv_folds"]

    X, X_train, X_test, y_train, y_test, median_exp = _split(cfg)
    n_feat = X.shape[1]
    print(f"train={X_train.shape} test={X_test.shape} features={n_feat}")

    Xs = X_train.sample(n=min(subset, len(X_train)), random_state=42)
    ys = y_train.loc[Xs.index]

    mlflow.set_tracking_uri(f"file:{ROOT / cfg['mlflow']['tracking_dir']}")
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    results = {}

    def record(name, pipe, extra=None):
        scores = cross_val_score(pipe, Xs, ys, cv=cv, scoring="r2", n_jobs=1)
        results[name] = {
            "pipe": pipe,
            "cv_r2": float(scores.mean()),
            "cv_r2_std": float(scores.std()),
        }
        with mlflow.start_run(run_name=name):
            mlflow.log_params(extra or {})
            mlflow.log_metrics({"cv_r2": float(scores.mean()), "cv_r2_std": float(scores.std())})
        print(f"{name:16} CV R2={scores.mean():.4f} +/- {scores.std():.4f}")

    # --- Baselines ---
    # alpha is small because the target is log-scaled (see build_pipeline).
    record("elasticnet", build_pipeline(ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), num, cat))
    rf = RandomForestRegressor(
        n_estimators=cfg["random_forest"]["n_estimators"],
        max_depth=cfg["random_forest"]["max_depth"],
        oob_score=True, random_state=42, n_jobs=-1,
    )
    record("random_forest", build_pipeline(rf, num, cat))
    record("xgboost", build_pipeline(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1), num, cat))
    record("catboost", build_pipeline(CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, random_seed=42, verbose=False, allow_writing_files=False), num, cat))

    # --- Optuna tuning on the same training-only subset ---
    xgb_space = {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": lambda t: t.suggest_int("max_depth", 4, 8),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.7, 1.0),
    }
    xgb_params, xgb_cv = _tune(
        lambda p: XGBRegressor(random_state=42, n_jobs=-1, **p), xgb_space, Xs, ys, num, cat, n_trials, cv
    )
    print(f"best xgb (cv R2={xgb_cv:.4f}): {xgb_params}")
    record("xgboost_tuned", build_pipeline(XGBRegressor(random_state=42, n_jobs=-1, **xgb_params), num, cat), xgb_params)

    cb_space = {
        "iterations": lambda t: t.suggest_int("iterations", 300, 700, step=100),
        "depth": lambda t: t.suggest_int("depth", 4, 8),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 6.0),
    }
    cb_params, cb_cv = _tune(
        lambda p: CatBoostRegressor(random_seed=42, verbose=False, **p, allow_writing_files=False), cb_space, Xs, ys, num, cat, n_trials, cv
    )
    print(f"best catboost (cv R2={cb_cv:.4f}): {cb_params}")
    record("catboost_tuned", build_pipeline(CatBoostRegressor(random_seed=42, verbose=False, **cb_params, allow_writing_files=False), num, cat), cb_params)

    # --- Stacking the two tuned learners ---
    base = [
        ("xgb", make_estimator(XGBRegressor(random_state=42, n_jobs=-1, **xgb_params), num, cat)),
        ("cb", make_estimator(CatBoostRegressor(random_seed=42, verbose=False, **cb_params, allow_writing_files=False), num, cat)),
    ]
    stack = StackingRegressor(estimators=base, final_estimator=RidgeCV(), cv=cv, n_jobs=1)
    record("stacking", TransformedTargetRegressor(regressor=stack, func=np.log1p, inverse_func=np.expm1))

    # --- Pick by CV, then touch the test set once ---
    best_name = select_best(results)
    best = results[best_name]
    best["pipe"].fit(X_train, y_train)
    best["metrics"] = regression_metrics(
        y_test, best["pipe"].predict(X_test), n_features=n_feat
    )
    print(
        f"\nWINNER BY CV: {best_name}  CV R2={best['cv_r2']:.4f}\n"
        f"FINAL TEST: R2={best['metrics']['r2']:.4f}  MAE={best['metrics']['mae']:,.0f}"
    )

    # --- Permutation importance on the winner ---
    # ponytail: one process avoids semaphore failures; parallelize if this becomes slow.
    perm = permutation_importance(
        best["pipe"], X_test, y_test, n_repeats=3 if fast else 5, random_state=42, n_jobs=1
    )
    top = sorted(zip(X.columns, perm.importances_mean), key=lambda kv: kv[1], reverse=True)[:10]
    print("top features:", [f"{c} ({v:.3f})" for c, v in top])

    # --- Export ---
    metadata = {
        "feature_columns": list(X.columns),
        "all_skills": cfg["skills"],
        "median_exp": median_exp,
        "model_name": best_name,
        "mae": float(best["metrics"]["mae"]),
        "rmse": float(best["metrics"]["rmse"]),
        "r2": float(best["metrics"]["r2"]),
        "mape": float(best["metrics"]["mape"]),
        "selection_cv_r2": float(best["cv_r2"]),
        "cv_results": {
            name: {"mean": result["cv_r2"], "std": result["cv_r2_std"]}
            for name, result in results.items()
        },
        "dataset_rows": len(X),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "job_titles": cfg["job_titles"],
        "locations": cfg["locations"],
        "education_levels": cfg["education_levels"],
    }
    model_path = ROOT / cfg["output"]["model"]
    metadata_path = ROOT / cfg["output"]["metadata"]
    for path in (model_path, metadata_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipe"], model_path)
    joblib.dump(metadata, metadata_path)
    with mlflow.start_run(run_name=f"{best_name}_final"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selection_cv_r2", best["cv_r2"])
        mlflow.log_metrics({f"test_{key}": value for key, value in best["metrics"].items()})
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")
    print(f"saved {cfg['output']['model']} and {cfg['output']['metadata']}")
    return results, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fast", action="store_true", help="tiny run for smoke testing")
    args = parser.parse_args()
    main(args.config, fast=args.fast)

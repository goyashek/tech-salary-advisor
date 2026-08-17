"""Train, tune, compare, and export the salary model.

Run: python -m src.train  (add --fast for a quick smoke run)

Steps: deterministic clean -> split -> train-fitted preprocessing -> model
comparison by cross-validation -> validation-only promotion -> final test
reporting -> registry/local fallback export. Runs use a local MLflow registry.
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from src.config import ROOT, load_config
from src.data import cap_outliers_iqr, clean, load_raw
from src.evaluate import (
    conformal_interval,
    conformal_quantile,
    interval_report,
    regression_metrics,
)
from src.features import build_features
from src.model_registry import (
    compare_candidate,
    configure_mlflow,
    get_champion_metrics,
    register_candidate,
)
from src.pipeline import build_pipeline, make_estimator
from src.validate_data import build_lineage, validate_data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _split(cfg):
    raw = load_raw(ROOT / cfg["data"]["path"])
    validation = validate_data(raw, target=cfg["data"]["target"])
    df = clean(raw, target=cfg["data"]["target"])
    df = build_features(df, cfg["skills"])
    y = df[cfg["data"]["target"]]
    X = df.drop(columns=[cfg["data"]["target"]])
    X_train_full, X_test, y_train_full_raw, y_test = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )
    X_train, X_validation, y_train_raw, y_validation = train_test_split(
        X_train_full,
        y_train_full_raw,
        test_size=cfg["split"].get("validation_size", 0.1),
        random_state=cfg["split"]["random_state"],
    )
    X_train, X_calibration, y_train_raw, y_calibration = train_test_split(
        X_train,
        y_train_raw,
        test_size=cfg["split"].get("calibration_size", 0.1),
        random_state=cfg["split"]["random_state"],
    )
    X_fit_full = pd.concat([X_train, X_validation])
    y_fit_full_raw = pd.concat([y_train_raw, y_validation])
    return {
        "X": X,
        "X_train": X_train,
        "X_fit_full": X_fit_full,
        "X_validation": X_validation,
        "X_calibration": X_calibration,
        "X_test": X_test,
        "y_train": cap_outliers_iqr(y_train_raw, cfg["outlier_iqr_factor"]),
        "y_fit_full": cap_outliers_iqr(y_fit_full_raw, cfg["outlier_iqr_factor"]),
        "y_validation": y_validation,
        "y_calibration": y_calibration,
        "y_test": y_test,
        "median_exp": float(X_fit_full["Experience_Years"].median()),
        "validation": validation,
    }


def _interval_segments(frame):
    experience = frame["Experience_Years"].to_numpy(dtype=float)
    buckets = np.full(len(frame), "missing", dtype=object)
    finite = np.isfinite(experience)
    buckets[finite & (experience <= 2)] = "0-2"
    buckets[finite & (experience > 2) & (experience <= 5)] = "3-5"
    buckets[finite & (experience > 5) & (experience <= 10)] = "6-10"
    buckets[finite & (experience > 10)] = "11+"
    titles = frame["Job_Title"].fillna("missing").astype(str).to_numpy()
    return {"experience_bucket": buckets, "job_title": titles}


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


def select_best(results, stacking_min_r2_gain=0.0):
    """Prefer one model unless stacking earns its configured complexity cost."""
    best_name = max(results, key=lambda name: results[name]["cv_r2"])
    if best_name != "stacking":
        return best_name
    best_single = max(
        (name for name in results if name != "stacking"),
        key=lambda name: results[name]["cv_r2"],
    )
    if (
        results["stacking"]["cv_r2"] - results[best_single]["cv_r2"]
        < stacking_min_r2_gain
    ):
        return best_single
    return best_name


def main(config_path="config.yaml", fast=False):
    import mlflow

    cfg = load_config(config_path)
    num, cat = cfg["numeric_features"], cfg["categorical_features"]
    n_trials = 2 if fast else cfg["tuning"]["n_trials"]
    subset = 3000 if fast else cfg["tuning"]["subset_size"]
    cv = cfg["tuning"]["cv_folds"]

    data = _split(cfg)
    X = data["X"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    interval_level = cfg.get("prediction_interval", {}).get("level", 0.9)
    validation = data["validation"]
    n_feat = X.shape[1]
    lineage = build_lineage(
        ROOT / cfg["data"]["path"],
        config_path,
        validation,
        cleaned_rows=len(X),
        training_rows=len(data["X_fit_full"]),
        validation_rows=len(data["X_validation"]),
        calibration_rows=len(data["X_calibration"]),
        test_rows=len(X_test),
        feature_count=n_feat,
    )
    print(
        f"train={X_train.shape} validation={data['X_validation'].shape} "
        f"calibration={data['X_calibration'].shape} "
        f"test={X_test.shape} features={n_feat}"
    )

    Xs = X_train.sample(n=min(subset, len(X_train)), random_state=42)
    ys = y_train.loc[Xs.index]

    tracking_uri = configure_mlflow(cfg, mlflow)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    results = {}

    def record(name, pipe, extra=None, explainability="medium"):
        scores = cross_val_score(pipe, Xs, ys, cv=cv, scoring="r2", n_jobs=1)
        results[name] = {
            "pipe": pipe,
            "cv_r2": float(scores.mean()),
            "cv_r2_std": float(scores.std()),
            "explainability": explainability,
        }
        with mlflow.start_run(run_name=name):
            mlflow.log_params(extra or {})
            mlflow.log_metrics(
                {"cv_r2": float(scores.mean()), "cv_r2_std": float(scores.std())}
            )
        print(f"{name:16} CV R2={scores.mean():.4f} +/- {scores.std():.4f}")

    # --- Baselines ---
    # alpha is small because the target is log-scaled (see build_pipeline).
    record(
        "elasticnet",
        build_pipeline(ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), num, cat),
        explainability="high",
    )
    rf = RandomForestRegressor(
        n_estimators=cfg["random_forest"]["n_estimators"],
        max_depth=cfg["random_forest"]["max_depth"],
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    record("random_forest", build_pipeline(rf, num, cat))
    record(
        "xgboost",
        build_pipeline(
            XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            ),
            num,
            cat,
        ),
    )
    record(
        "catboost",
        build_pipeline(
            CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
            num,
            cat,
        ),
    )
    record(
        "mlp",
        build_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                max_iter=1000,
                random_state=42,
            ),
            num,
            cat,
        ),
        {
            "hidden_layer_sizes": "64,32",
            "early_stopping": True,
            "n_iter_no_change": 30,
            "max_iter": 1000,
        },
        explainability="low",
    )

    # --- Optuna tuning on the same training-only subset ---
    xgb_space = {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": lambda t: t.suggest_int("max_depth", 4, 8),
        "learning_rate": lambda t: t.suggest_float(
            "learning_rate", 0.03, 0.2, log=True
        ),
        "subsample": lambda t: t.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.7, 1.0),
    }
    xgb_params, xgb_cv = _tune(
        lambda p: XGBRegressor(random_state=42, n_jobs=-1, **p),
        xgb_space,
        Xs,
        ys,
        num,
        cat,
        n_trials,
        cv,
    )
    print(f"best xgb (cv R2={xgb_cv:.4f}): {xgb_params}")
    record(
        "xgboost_tuned",
        build_pipeline(
            XGBRegressor(random_state=42, n_jobs=-1, **xgb_params), num, cat
        ),
        xgb_params,
    )

    cb_space = {
        "iterations": lambda t: t.suggest_int("iterations", 300, 700, step=100),
        "depth": lambda t: t.suggest_int("depth", 4, 8),
        "learning_rate": lambda t: t.suggest_float(
            "learning_rate", 0.03, 0.15, log=True
        ),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 6.0),
    }
    cb_params, cb_cv = _tune(
        lambda p: CatBoostRegressor(
            random_seed=42, verbose=False, **p, allow_writing_files=False
        ),
        cb_space,
        Xs,
        ys,
        num,
        cat,
        n_trials,
        cv,
    )
    print(f"best catboost (cv R2={cb_cv:.4f}): {cb_params}")
    record(
        "catboost_tuned",
        build_pipeline(
            CatBoostRegressor(
                random_seed=42, verbose=False, **cb_params, allow_writing_files=False
            ),
            num,
            cat,
        ),
        cb_params,
    )

    # --- Stacking the two tuned learners ---
    base = [
        (
            "xgb",
            make_estimator(
                XGBRegressor(random_state=42, n_jobs=-1, **xgb_params), num, cat
            ),
        ),
        (
            "cb",
            make_estimator(
                CatBoostRegressor(
                    random_seed=42,
                    verbose=False,
                    **cb_params,
                    allow_writing_files=False,
                ),
                num,
                cat,
            ),
        ),
    ]
    stack = StackingRegressor(
        estimators=base, final_estimator=RidgeCV(), cv=cv, n_jobs=1
    )
    record(
        "stacking",
        TransformedTargetRegressor(
            regressor=stack, func=np.log1p, inverse_func=np.expm1
        ),
        explainability="low",
    )

    # --- Pick by CV, then compare on the held-out promotion split ---
    best_single = max(
        (name for name in results if name != "stacking"),
        key=lambda name: results[name]["cv_r2"],
    )
    stacking_gain = results["stacking"]["cv_r2"] - results[best_single]["cv_r2"]
    best_name = select_best(results, cfg["tuning"]["stacking_min_r2_gain"])
    best = results[best_name]
    best["pipe"].fit(X_train, y_train)
    promotion_metrics = regression_metrics(
        data["y_validation"],
        best["pipe"].predict(data["X_validation"]),
        n_features=n_feat,
    )
    champion = get_champion_metrics(
        client,
        cfg["mlflow"]["registered_model"],
        cfg["mlflow"]["champion_alias"],
    )
    promotion = compare_candidate(
        {"mae": promotion_metrics["mae"], "r2": promotion_metrics["r2"]},
        champion,
        **cfg["mlflow"]["promotion"],
    )

    # Refit without calibration rows, then use their untouched residuals for intervals.
    best["pipe"].fit(data["X_fit_full"], data["y_fit_full"])
    calibration_predictions = best["pipe"].predict(data["X_calibration"])
    interval_half_width = conformal_quantile(
        data["y_calibration"], calibration_predictions, level=interval_level
    )
    calibration_lower, calibration_upper = conformal_interval(
        calibration_predictions, interval_half_width
    )
    calibration_interval = interval_report(
        data["y_calibration"].to_numpy(),
        calibration_lower,
        calibration_upper,
        _interval_segments(data["X_calibration"]),
    )
    test_predictions = best["pipe"].predict(X_test)
    test_lower, test_upper = conformal_interval(test_predictions, interval_half_width)
    test_interval = interval_report(
        y_test.to_numpy(),
        test_lower,
        test_upper,
        _interval_segments(X_test),
    )
    best["metrics"] = regression_metrics(y_test, test_predictions, n_features=n_feat)
    print(
        f"\nWINNER BY CV: {best_name}  CV R2={best['cv_r2']:.4f}\n"
        f"STACKING GAIN over {best_single}: {stacking_gain:+.4f}\n"
        f"PROMOTION: {promotion['decision']}  validation R2={promotion_metrics['r2']:.4f} "
        f"MAE={promotion_metrics['mae']:,.0f}\n"
        f"FINAL TEST: R2={best['metrics']['r2']:.4f}  MAE={best['metrics']['mae']:,.0f}\n"
        f"90% INTERVAL: half-width=₹{interval_half_width:,.0f} "
        f"calibration coverage={calibration_interval['coverage']:.3f}"
    )

    # --- Permutation importance on the winner ---
    # ponytail: one process avoids semaphore failures; parallelize if this becomes slow.
    perm = permutation_importance(
        best["pipe"],
        X_test,
        y_test,
        n_repeats=3 if fast else 5,
        random_state=42,
        n_jobs=1,
    )
    top = sorted(
        zip(X.columns, perm.importances_mean), key=lambda kv: kv[1], reverse=True
    )[:10]
    print("top features:", [f"{c} ({v:.3f})" for c, v in top])

    # --- Export ---
    metadata = {
        "feature_columns": list(X.columns),
        "all_skills": cfg["skills"],
        "median_exp": data["median_exp"],
        "model_name": best_name,
        "mae": float(best["metrics"]["mae"]),
        "rmse": float(best["metrics"]["rmse"]),
        "r2": float(best["metrics"]["r2"]),
        "mape": float(best["metrics"]["mape"]),
        "selection_cv_r2": float(best["cv_r2"]),
        "cv_results": {
            name: {
                "mean": result["cv_r2"],
                "std": result["cv_r2_std"],
                "explainability": result["explainability"],
            }
            for name, result in results.items()
        },
        "stacking_comparison": {
            "best_single_model": best_single,
            "best_single_cv_r2": results[best_single]["cv_r2"],
            "stacking_cv_r2": results["stacking"]["cv_r2"],
            "stacking_gain": stacking_gain,
            "minimum_required_gain": cfg["tuning"]["stacking_min_r2_gain"],
        },
        "dataset_rows": len(X),
        "training_rows": len(data["X_fit_full"]),
        "validation_rows": len(data["X_validation"]),
        "calibration_rows": len(data["X_calibration"]),
        "test_rows": len(X_test),
        "job_titles": cfg["job_titles"],
        "locations": cfg["locations"],
        "education_levels": cfg["education_levels"],
        "data_validation": validation,
        "lineage": lineage,
        "promotion": promotion,
        "promotion_metrics": promotion_metrics,
        "prediction_interval": {
            "method": "split_conformal",
            "level": float(interval_level),
            "quantile_inr": interval_half_width,
            "calibration": calibration_interval,
            "final_test": test_interval,
        },
    }
    model_path = ROOT / cfg["output"]["model"]
    metadata_path = ROOT / cfg["output"]["metadata"]
    with mlflow.start_run(run_name=f"{best_name}_candidate"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_params(lineage)
        mlflow.log_metric("selection_cv_r2", best["cv_r2"])
        mlflow.log_metrics(
            {f"validation_{key}": value for key, value in promotion_metrics.items()}
        )
        mlflow.log_metrics(
            {f"test_{key}": value for key, value in best["metrics"].items()}
        )
        mlflow.log_metrics(
            {
                "interval_level": interval_level,
                "interval_half_width_inr": interval_half_width,
                "calibration_interval_coverage": calibration_interval["coverage"],
                "calibration_interval_mean_width": calibration_interval["mean_width"],
                "test_interval_coverage": test_interval["coverage"],
                "test_interval_mean_width": test_interval["mean_width"],
            }
        )
        mlflow.log_dict(validation, "data_validation.json")
        mlflow.log_dict(lineage, "lineage.json")
        mlflow.log_dict(promotion, "promotion.json")
        mlflow.log_dict(metadata["prediction_interval"], "prediction_interval.json")
        model_info = mlflow.sklearn.log_model(
            best["pipe"],
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        version = register_candidate(
            mlflow,
            client,
            model_info.model_uri,
            cfg["mlflow"]["registered_model"],
            promotion["candidate"],
            promotion,
            lineage,
            cfg["mlflow"]["challenger_alias"],
            cfg["mlflow"]["champion_alias"],
        )
        metadata["registry_version"] = str(version.version)
        mlflow.log_dict(metadata, "model/metadata.json")

    if promotion["decision"] == "promote":
        for path in (model_path, metadata_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["pipe"], model_path)
        joblib.dump(metadata, metadata_path)
        print(f"promoted registry version {version.version}; saved local fallback")
    else:
        print("candidate rejected; existing champion and local fallback remain")
    return results, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--fast", action="store_true", help="tiny run for smoke testing"
    )
    args = parser.parse_args()
    main(args.config, fast=args.fast)

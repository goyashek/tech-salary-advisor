"""Training-only Keras ANN benchmark for the salary regression task.

Run on Apple Silicon with: .venv/bin/python -m src.keras_benchmark
TensorFlow uses the Metal GPU automatically when it is available.
"""

import argparse

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from src.config import ROOT, load_config
from src.data import cap_outliers_iqr, clean, load_raw
from src.features import build_features
from src.pipeline import build_preprocessor
from src.validate_data import validate_data

SEED = 42


def _training_subset(cfg, subset_size):
    """Recreate the existing training/promotion/test split without using test labels."""
    raw = load_raw(ROOT / cfg["data"]["path"])
    validate_data(raw, target=cfg["data"]["target"])
    df = build_features(clean(raw, target=cfg["data"]["target"]), cfg["skills"])
    y = df[cfg["data"]["target"]]
    X = df.drop(columns=[cfg["data"]["target"]])
    X_train_full, _, y_train_full, _ = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )
    X_train, _, y_train, _ = train_test_split(
        X_train_full,
        y_train_full,
        test_size=cfg["split"]["validation_size"],
        random_state=cfg["split"]["random_state"],
    )
    y_train = cap_outliers_iqr(y_train, cfg["outlier_iqr_factor"])
    X_sample = X_train.sample(n=min(subset_size, len(X_train)), random_state=SEED)
    return X_sample, y_train.loc[X_sample.index]


def build_ann(input_dim):
    """Build one regularized dense network for tabular regression."""
    import keras

    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(256, use_bias=False, kernel_initializer="he_normal"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(128, use_bias=False, kernel_initializer="he_normal"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(64, use_bias=False, kernel_initializer="he_normal"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.Dense(1),
        ],
        name="salary_ann",
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0
        ),
        loss="mean_squared_error",
    )
    return model


def run_benchmark(config_path="config.yaml", fast=False):
    import keras
    import tensorflow as tf

    cfg = load_config(config_path)
    subset_size = 3000 if fast else cfg["tuning"]["subset_size"]
    epochs = 30 if fast else 300
    patience = 5 if fast else 20
    X, y = _training_subset(cfg, subset_size)
    cv = KFold(n_splits=cfg["tuning"]["cv_folds"])
    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"device={device} rows={len(X)} folds={cv.n_splits}")

    fold_results = []
    for fold, (train_idx, validation_idx) in enumerate(cv.split(X), start=1):
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + fold)
        preprocessor = build_preprocessor(
            cfg["numeric_features"], cfg["categorical_features"]
        )
        X_train = preprocessor.fit_transform(X.iloc[train_idx]).astype("float32")
        X_validation = preprocessor.transform(X.iloc[validation_idx]).astype("float32")
        y_train_log = np.log1p(y.iloc[train_idx].to_numpy(dtype="float32"))
        y_validation_log = np.log1p(y.iloc[validation_idx].to_numpy(dtype="float32"))
        target_mean, target_std = y_train_log.mean(), y_train_log.std()
        y_train = (y_train_log - target_mean) / target_std
        y_validation = (y_validation_log - target_mean) / target_std

        model = build_ann(X_train.shape[1])
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
            epochs=epochs,
            batch_size=256,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                ),
            ],
            verbose=0,
        )
        predictions = np.expm1(
            model.predict(X_validation, verbose=0).reshape(-1) * target_std
            + target_mean
        )
        actual = y.iloc[validation_idx].to_numpy()
        result = {
            "r2": float(r2_score(actual, predictions)),
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(root_mean_squared_error(actual, predictions)),
            "epochs": len(history.history["loss"]),
        }
        fold_results.append(result)
        print(
            f"fold={fold} r2={result['r2']:.4f} mae={result['mae']:,.0f} "
            f"epochs={result['epochs']}"
        )

    summary = {
        "device": device,
        "cv_r2": float(np.mean([result["r2"] for result in fold_results])),
        "cv_r2_std": float(np.std([result["r2"] for result in fold_results])),
        "cv_mae": float(np.mean([result["mae"] for result in fold_results])),
        "folds": fold_results,
    }
    print(
        f"keras_ann CV R2={summary['cv_r2']:.4f} +/- {summary['cv_r2_std']:.4f} "
        f"MAE={summary['cv_mae']:,.0f}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.config, args.fast)

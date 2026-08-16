.PHONY: install install-dev test train train-fast keras-benchmark app mlflow clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q

# Full training run: tunes, compares, and exports the best model.
train:
	python -m src.train

# Quick end-to-end check (tiny Optuna budget), handy for CI.
train-fast:
	python -m src.train --fast

keras-benchmark:
	.venv/bin/python -m src.keras_benchmark

app:
	streamlit run streamlit/app.py

# Browse the logged experiments.
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf mlruns mlflow.db .pytest_cache __pycache__ src/__pycache__ tests/__pycache__

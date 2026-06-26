.PHONY: install test train train-fast app mlflow lint clean

install:
	pip install -r requirements.txt

test:
	pytest -q

# Full training run: tunes, compares, and exports the best model.
train:
	python -m src.train

# Quick end-to-end check (tiny Optuna budget), handy for CI.
train-fast:
	python -m src.train --fast

app:
	streamlit run streamlit/app.py

# Browse the logged experiments.
mlflow:
	MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns

clean:
	rm -rf mlruns .pytest_cache __pycache__ src/__pycache__ tests/__pycache__

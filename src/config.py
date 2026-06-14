"""Load the YAML config as a plain dict."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="config.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)

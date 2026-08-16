"""Small MLflow registry helpers for champion/challenger promotion."""

from src.config import ROOT


def resolve_tracking_uri(uri):
    """Resolve the configured local SQLite database against the repo root."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith("sqlite:////"):
        return f"sqlite:///{(ROOT / uri[len(prefix) :]).resolve()}"
    return uri


def configure_mlflow(cfg, mlflow):
    """Point MLflow tracking and registry at the same configured backend."""
    uri = resolve_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def compare_candidate(
    candidate,
    champion,
    max_mae_ratio=1.02,
    min_r2_delta=-0.005,
):
    """Apply the explicit validation-only promotion rule."""
    thresholds = {
        "max_mae_ratio": float(max_mae_ratio),
        "min_r2_delta": float(min_r2_delta),
    }
    if champion is None:
        return {
            "decision": "promote",
            "reason": "no champion exists",
            "candidate": candidate,
            "champion": None,
            "thresholds": thresholds,
            "checks": {"mae": True, "r2": True},
        }

    checks = {
        "mae": candidate["mae"] <= champion["mae"] * thresholds["max_mae_ratio"],
        "r2": candidate["r2"] >= champion["r2"] + thresholds["min_r2_delta"],
    }
    return {
        "decision": "promote" if all(checks.values()) else "reject",
        "reason": "all promotion checks passed"
        if all(checks.values())
        else "metric gate failed",
        "candidate": candidate,
        "champion": champion,
        "thresholds": thresholds,
        "checks": checks,
    }


def get_champion_metrics(client, model_name, alias):
    """Read validation metrics from the current champion alias."""
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None
    try:
        mae = float(version.tags["validation_mae"])
        r2 = float(version.tags["validation_r2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("champion is missing validation metrics") from exc
    return {"version": str(version.version), "mae": mae, "r2": r2}


def register_candidate(
    mlflow,
    client,
    model_uri,
    model_name,
    validation_metrics,
    decision,
    lineage,
    challenger_alias,
    champion_alias,
):
    """Register a candidate, expose it as challenger, and promote when allowed."""
    version = mlflow.register_model(model_uri, model_name)
    version_number = str(version.version)
    tags = {
        "promotion_status": decision["decision"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "git_sha": lineage["git_sha"],
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version_number, key, str(value))
    client.set_registered_model_alias(model_name, challenger_alias, version_number)
    if decision["decision"] == "promote":
        client.set_registered_model_alias(model_name, champion_alias, version_number)
    return version

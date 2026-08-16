import mlflow
from sklearn.dummy import DummyRegressor

from src.inference import _load_registry_assets
from src.model_registry import (
    compare_candidate,
    get_champion_metrics,
    register_candidate,
)


def test_weaker_candidate_is_rejected():
    decision = compare_candidate(
        {"mae": 110.0, "r2": 0.80},
        {"version": "1", "mae": 100.0, "r2": 0.81},
    )

    assert decision["decision"] == "reject"
    assert decision["checks"] == {"mae": False, "r2": False}


def test_candidate_can_pass_the_promotion_gate():
    decision = compare_candidate(
        {"mae": 101.0, "r2": 0.806},
        {"version": "1", "mae": 100.0, "r2": 0.81},
    )

    assert decision["decision"] == "promote"


def test_registry_keeps_champion_and_serves_promoted_model(tmp_path):
    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment("registry-test")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    cfg = {
        "mlflow": {
            "tracking_uri": uri,
            "registered_model": "registry-test-model",
            "champion_alias": "champion",
        }
    }

    def log_candidate(metrics, decision):
        model = DummyRegressor(strategy="mean").fit([[0], [1]], [1, 2])
        with mlflow.start_run():
            model_info = mlflow.sklearn.log_model(
                model, artifact_path="model", serialization_format="cloudpickle"
            )
            mlflow.log_dict({"model_name": "test"}, "model/metadata.json")
            return register_candidate(
                mlflow,
                client,
                model_info.model_uri,
                cfg["mlflow"]["registered_model"],
                metrics,
                decision,
                {"git_sha": "test"},
                "challenger",
                cfg["mlflow"]["champion_alias"],
            )

    first_metrics = {"mae": 100.0, "r2": 0.80}
    first = log_candidate(first_metrics, compare_candidate(first_metrics, None))
    assert get_champion_metrics(client, cfg["mlflow"]["registered_model"], "champion")[
        "version"
    ] == str(first.version)

    rejected_metrics = {"mae": 130.0, "r2": 0.70}
    rejected = log_candidate(
        rejected_metrics,
        compare_candidate(
            rejected_metrics, {**first_metrics, "version": str(first.version)}
        ),
    )
    assert (
        client.get_model_version_by_alias(
            cfg["mlflow"]["registered_model"], "champion"
        ).version
        == first.version
    )
    assert (
        client.get_model_version_by_alias(
            cfg["mlflow"]["registered_model"], "challenger"
        ).version
        == rejected.version
    )

    promoted_metrics = {"mae": 95.0, "r2": 0.81}
    promoted = log_candidate(
        promoted_metrics,
        compare_candidate(
            promoted_metrics,
            {**first_metrics, "version": str(first.version)},
        ),
    )
    assert (
        client.get_model_version_by_alias(
            cfg["mlflow"]["registered_model"], "champion"
        ).version
        == promoted.version
    )

    model, metadata = _load_registry_assets(cfg)
    assert metadata == {"model_name": "test"}
    assert model.predict([[0]])[0] == 1.5

"""Tests du registre de modèles (données fabriquées, base SQLite temporaire).

Ils vérifient la logique du registre, pas la qualité du modèle.
"""
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import registry as reg

EXPERIMENT = "test-registry"


@pytest.fixture()
def client(tmp_path):
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(
        EXPERIMENT, artifact_location=(tmp_path / "art").as_uri()
    )
    mlflow.set_experiment(EXPERIMENT)
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


def _model():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 3)
    y = (X[:, 0] > 0).astype(int)
    return Pipeline(
        [("s", StandardScaler()), ("c", LogisticRegression())]
    ).fit(X, y)


def _log_run(stage, sample_frac, auc, f1=0.44):
    """Reproduit ce que journalise train.py (noms et formats)."""
    with mlflow.start_run(run_name=f"{stage}-{sample_frac}") as run:
        mlflow.log_params(
            {
                "model": "hist_gradient_boosting",
                "feature_set": "core",
                "add_ratios": False,
                "class_weight": None,
                "seed": 42,
                "sample_frac": sample_frac,
                "n_fit": 400000,
            }
        )
        mlflow.set_tag("stage", stage)
        mlflow.log_metrics(
            {reg.PRIMARY_METRIC: auc, reg.SECONDARY_METRIC: f1}
        )
        try:
            mlflow.sklearn.log_model(_model(), name="model")
        except TypeError:
            mlflow.sklearn.log_model(_model(), artifact_path="model")
    return run.info.run_id


def test_find_final_run_ignores_sample_and_ablation(client):
    _log_run("ablation", None, 0.60)
    sample_id = _log_run("final", 0.05, 0.66)
    full_id = _log_run("final", None, 0.675)
    run = reg.find_final_run(client, EXPERIMENT)
    assert run.info.run_id == full_id
    assert run.info.run_id != sample_id


def test_find_final_run_takes_latest_full_run(client):
    _log_run("final", None, 0.670)
    latest = _log_run("final", None, 0.680)
    assert reg.find_final_run(client, EXPERIMENT).info.run_id == latest


def test_find_final_run_raises_without_full_run(client):
    _log_run("final", 0.05, 0.66)
    with pytest.raises(ValueError, match="jeu complet"):
        reg.find_final_run(client, EXPERIMENT)


def test_find_final_run_unknown_experiment(client):
    with pytest.raises(ValueError, match="introuvable"):
        reg.find_final_run(client, "n-existe-pas")


def test_register_sets_staging_alias_tags_and_description(client):
    run = client.get_run(_log_run("final", None, 0.675))
    version = reg.register_candidate(client, run, "m-test")
    assert STAGING_IN(version)
    tags = version.tags
    assert tags["feature_set"] == "core"
    assert tags["run_id"] == run.info.run_id
    assert tags[reg.PRIMARY_METRIC] == "0.6750"
    assert "impayé" in version.description


def STAGING_IN(version):
    return reg.STAGING in version.aliases


def test_register_is_idempotent(client):
    run = client.get_run(_log_run("final", None, 0.675))
    first = reg.register_candidate(client, run, "m-test")
    second = reg.register_candidate(client, run, "m-test")
    assert str(first.version) == str(second.version)
    assert len(client.search_model_versions("name='m-test'")) == 1


@pytest.mark.parametrize(
    "cand, prod, delta, expected",
    [
        (0.675, None, 0.0, True),
        (0.680, 0.675, 0.0, True),
        (0.675, 0.675, 0.0, True),
        (0.670, 0.675, 0.0, False),
        (0.676, 0.675, 0.005, False),
        (0.681, 0.675, 0.005, True),
    ],
)
def test_decide(cand, prod, delta, expected):
    production = None if prod is None else {reg.PRIMARY_METRIC: prod}
    ok, reason = reg.decide({reg.PRIMARY_METRIC: cand}, production, delta)
    assert ok is expected
    assert reason


def test_decide_missing_metric():
    with pytest.raises(KeyError):
        reg.decide({}, None)
    with pytest.raises(KeyError):
        reg.decide({reg.PRIMARY_METRIC: 0.7}, {"autre": 1.0})


def test_promote_keeps_previous_and_rollback(client):
    r1 = client.get_run(_log_run("final", None, 0.670))
    v1 = reg.register_candidate(client, r1, "m-test")
    reg.promote(client, "m-test", v1.version)
    r2 = client.get_run(_log_run("final", None, 0.680))
    v2 = reg.register_candidate(client, r2, "m-test")
    reg.promote(client, "m-test", v2.version)

    prod = reg.get_by_alias(client, "m-test", reg.PRODUCTION)
    prev = reg.get_by_alias(client, "m-test", reg.PREVIOUS)
    assert str(prod.version) == str(v2.version)
    assert str(prev.version) == str(v1.version)

    restored = reg.rollback(client, "m-test")
    assert restored == str(v1.version)
    current = reg.get_by_alias(client, "m-test", reg.PRODUCTION)
    assert str(current.version) == str(v1.version)


def test_rollback_without_previous_raises(client):
    r1 = client.get_run(_log_run("final", None, 0.670))
    reg.register_candidate(client, r1, "m-test")
    with pytest.raises(ValueError, match="rollback impossible"):
        reg.rollback(client, "m-test")


def test_get_by_alias_missing_returns_none(client):
    assert reg.get_by_alias(client, "inconnu", reg.PRODUCTION) is None


def test_load_production_model_predicts(client):
    run = client.get_run(_log_run("final", None, 0.675))
    version = reg.register_candidate(client, run, "m-test")
    reg.promote(client, "m-test", version.version)
    model = reg.load_production_model("m-test")
    X = np.random.RandomState(1).randn(5, 3)
    assert model.predict(X).shape == (5,)


def test_cli_registers_then_refuses_worse_candidate(
    client, tmp_path, monkeypatch, capsys
):
    params = tmp_path / "params.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "mlflow": {
                    "tracking_uri": mlflow.get_tracking_uri(),
                    "experiment_name": EXPERIMENT,
                }
            }
        ),
        encoding="utf-8",
    )
    base = ["registry", "--params", str(params), "--name", "m-cli"]

    _log_run("final", None, 0.675)
    monkeypatch.setattr(sys, "argv", base)
    reg.main()
    assert "PROMOTION" in capsys.readouterr().out

    _log_run("final", None, 0.640)
    monkeypatch.setattr(sys, "argv", base)
    reg.main()
    assert "REFUS" in capsys.readouterr().out
    prod = reg.get_by_alias(MlflowClient(), "m-cli", reg.PRODUCTION)
    assert str(prod.version) == "1"


def test_promote_same_version_twice_sets_no_previous(client):
    run = client.get_run(_log_run("final", None, 0.675))
    version = reg.register_candidate(client, run, "m-test")
    reg.promote(client, "m-test", version.version)
    reg.promote(client, "m-test", version.version)
    assert reg.get_by_alias(client, "m-test", reg.PREVIOUS) is None


def test_show_displays_aliases(client, capsys):
    r1 = client.get_run(_log_run("final", None, 0.670))
    v1 = reg.register_candidate(client, r1, "m-test")
    reg.promote(client, "m-test", v1.version)
    r2 = client.get_run(_log_run("final", None, 0.680))
    reg.register_candidate(client, r2, "m-test")
    reg.show(client, "m-test")
    lines = capsys.readouterr().out.strip().splitlines()
    assert "alias=production" in lines[0]
    assert "alias=staging" in lines[1]

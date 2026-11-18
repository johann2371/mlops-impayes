"""Tests de src/models (tâche 5).

Les données sont **fabriquées** (schéma réel, signal artificiel fort sur
``late_payment_count``) : elles vérifient le code, pas la qualité du
modèle sur le vrai jeu.
"""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf
from src.models import evaluate as ev
from src.models import train


def make_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    late = rng.integers(0, 11, n)
    tenure = rng.integers(1, 121, n)
    score = late * 3 - tenure / 40 + rng.normal(0, 1, n)
    risk = np.where(score > 7, "high", np.where(score > 3, "medium", "low"))
    return pd.DataFrame(
        {
            "customer_id": [f"CUST{i}" for i in range(n)],
            "phone_number": 2347000000000 + np.arange(n),
            "operator": rng.choice(["MTN", "Airtel", "Glo", "9mobile"], n),
            "assessment_date": "2025-09-30",
            "credit_score": rng.integers(300, 1001, n),
            "credit_rating": rng.choice(
                ["poor", "fair", "good", "excellent"], n
            ),
            "tenure_months": tenure,
            "avg_monthly_spend_ngn": rng.uniform(500, 50000, n).round(2),
            "recharge_frequency_per_month": rng.integers(1, 31, n),
            "payment_consistency_score": rng.integers(0, 101, n) / 100,
            "late_payment_count": late,
            "account_type": rng.choice(["prepaid", "postpaid"], n),
            "loan_eligibility": rng.uniform(size=n) > 0.3,
            "max_loan_amount_ngn": rng.uniform(0, 150000, n).round(2),
            "default_risk": risk,
        }
    )


@pytest.fixture
def params(tmp_path):
    p = train.load_params("params.yaml")
    p["data"]["raw_path"] = str(tmp_path / "raw.csv")
    p["model"]["name"] = "logistic_regression"
    p["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    p["mlflow"]["artifact_location"] = str(tmp_path / "artifacts")
    p["outputs"] = {
        "model_path": str(tmp_path / "m" / "model.joblib"),
        "model_meta_path": str(tmp_path / "m" / "meta.json"),
        "metrics_path": str(tmp_path / "r" / "metrics.json"),
        "confusion_matrix_path": str(tmp_path / "r" / "cm.png"),
        "ablation_path": str(tmp_path / "r" / "ablation.csv"),
    }
    return p


def test_compute_metrics_perfect_and_constant():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y]
    m = ev.compute_metrics(y, y, proba)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["roc_auc_ovr_macro"] == pytest.approx(1.0)
    assert ev.confusion(y, y).trace() == len(y)
    assert ev.baseline_f1_macro(y, y) == pytest.approx(0.2, abs=0.05)


def test_confusion_matrix_shape_and_counts():
    y_true = np.array([0, 0, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])
    cm = ev.confusion(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)
    assert cm[2, 0] == 1


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="inconnu"):
        train.build_model("foret_magique", None, 0, {})


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "hist_gradient_boosting"]
)
def test_final_run_saves_usable_model(params, model_name):
    params["model"]["name"] = model_name
    params["model"]["hist_gradient_boosting"]["max_iter"] = 20
    df = make_df()
    df.to_csv(params["data"]["raw_path"], index=False)
    res = train.run_experiment(params, train.load_dataset(params))
    assert res["test_f1_macro"] > res["test_baseline_f1_macro"]
    assert np.array(res["confusion_matrix"]).shape == (3, 3)

    model = joblib.load(params["outputs"]["model_path"])
    meta = json.loads(open(params["outputs"]["model_meta_path"]).read())
    assert meta["classes"] == ["low", "medium", "high"]
    X_new = df[meta["input_columns"]].head(5)
    proba = model.predict_proba(X_new)
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert bf.decode_target(model.predict(X_new))


def test_ablation_uses_validation_not_test(params):
    df = make_df(n=1500)
    p = copy.deepcopy(params)
    table = train.run_ablation(p, df)
    assert len(table) == 8
    assert "val_f1_macro" in table.columns
    assert not any(c.startswith("test_") for c in table.columns)
    assert table["val_f1_macro"].is_monotonic_decreasing


def test_unknown_evaluate_on_raises(params):
    with pytest.raises(ValueError):
        train.run_experiment(params, make_df(300), evaluate_on="autre")

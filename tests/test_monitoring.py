"""Tests du monitoring (données fabriquées, aucun accès au vrai jeu).

Ils vérifient la logique de détection et d'alerte, pas la qualité du
modèle ni les valeurs sur le vrai jeu.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.monitoring import drift
from src.monitoring import simulate_drift as sim

DASHBOARD = Path(__file__).resolve().parents[1] / "src/monitoring/dashboard.py"


# ------------------------------------------------------------ fabrique
def make_data(n=6000, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "avg_monthly_spend_ngn": rng.uniform(1000, 50000, n),
            "recharge_frequency_per_month": rng.uniform(1, 30, n),
            "late_payment_count": rng.randint(0, 11, n),
            "operator": rng.choice(["a", "b", "c"], n),
        }
    )
    score = (
        X["late_payment_count"] / 10
        - X["recharge_frequency_per_month"] / 30
        + rng.normal(0, 0.3, n)
    )
    y = np.digitize(score, np.quantile(score, [0.5, 0.75]))
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = make_data()
    X_ref, X_test = X.iloc[:4000], X.iloc[4000:]
    y_ref, y_test = y[:4000], y[4000:]
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [("oh", OneHotEncoder(), ["operator"])],
                    remainder="passthrough",
                ),
            ),
            ("clf", HistGradientBoostingClassifier(random_state=0)),
        ]
    ).fit(X_ref, y_ref)
    return model, X_ref, X_test, y_test


# ----------------------------------------------------------------- PSI
def test_psi_identical_is_zero():
    s = pd.Series(np.random.RandomState(0).randn(5000))
    assert drift.psi(s, s) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_shift():
    rng = np.random.RandomState(0)
    assert drift.psi(rng.randn(5000), rng.randn(5000) + 1.5) > 0.25


def test_psi_same_distribution_is_small():
    rng = np.random.RandomState(1)
    assert drift.psi(rng.randn(20000), rng.randn(20000)) < 0.05


def test_psi_categorical_and_unseen_category():
    ref = pd.Series(["a", "b"] * 500)
    cur = pd.Series(["a", "b", "c"] * 300)
    value = drift.psi(ref, cur)
    assert math.isfinite(value) and value > 0.1


def test_psi_discrete_numeric_uses_modalities():
    ref = pd.Series(np.random.RandomState(0).randint(0, 11, 5000))
    cur = pd.Series(np.full(5000, 10))
    assert drift.psi(ref, cur) > 0.25


def test_psi_empty_raises():
    with pytest.raises(ValueError):
        drift.psi(pd.Series([], dtype=float), pd.Series([1.0]))


def test_prediction_psi():
    same = np.array([0] * 50 + [1] * 25 + [2] * 25)
    assert drift.prediction_psi(same, same) == pytest.approx(0.0, abs=1e-9)
    assert drift.prediction_psi(same, np.zeros(100, dtype=int)) > 0.25


# --------------------------------------------------------- performance
def test_performance_metrics_perfect_and_missing_class():
    y = np.array([0, 1, 2] * 10)
    proba = np.eye(3)[y]
    m = drift.performance_metrics(y, y, proba)
    assert m["f1_macro"] == 1.0 and m["roc_auc_ovr_macro"] == 1.0
    only_two = np.array([0, 1] * 15)
    m2 = drift.performance_metrics(only_two, only_two, np.eye(3)[only_two])
    assert m2["roc_auc_ovr_macro"] is None


# -------------------------------------------------------------- alertes
BASE = {"f1_macro": 0.45, "roc_auc_ovr_macro": 0.675}


def test_build_alerts_levels():
    alerts = drift.build_alerts(
        {"a": 0.05, "b": 0.15, "c": 0.40}, 0.0,
        {"f1_macro": 0.45, "roc_auc_ovr_macro": 0.675}, BASE,
    )
    by_name = {a["name"]: a["level"] for a in alerts}
    assert "a" not in by_name
    assert by_name["b"] == "warning"
    assert by_name["c"] == "critical"
    assert drift.batch_status(alerts) == "critical"


def test_build_alerts_performance_drop():
    alerts = drift.build_alerts(
        {}, None, {"f1_macro": 0.40, "roc_auc_ovr_macro": 0.60}, BASE
    )
    levels = {a["name"]: a["level"] for a in alerts}
    assert levels["f1_macro"] == "critical"
    assert levels["roc_auc_ovr_macro"] == "critical"


def test_build_alerts_ignores_missing_metrics_and_baseline():
    assert drift.build_alerts({}, None, None, BASE) == []
    assert drift.build_alerts(
        {}, None, {"f1_macro": 0.1, "roc_auc_ovr_macro": None}, None
    ) == []
    assert drift.build_alerts(
        {}, None, {"f1_macro": None, "roc_auc_ovr_macro": None}, BASE
    ) == []


def test_build_alerts_custom_thresholds():
    alerts = drift.build_alerts(
        {"a": 0.05}, None, None, None, {"psi_warning": 0.01}
    )
    assert alerts and alerts[0]["level"] == "warning"


def test_status_ok_when_no_alert():
    assert drift.batch_status([]) == "ok"


def test_retrain_recommended_only_on_performance_drop():
    perf = drift.build_alerts(
        {}, None, {"f1_macro": 0.3, "roc_auc_ovr_macro": 0.675}, BASE
    )
    d = drift.retrain_decision(perf)
    assert d["recommended"] and d["automatic"] is False
    assert not d["investigate"]
    assert not drift.retrain_decision([])["recommended"]


def test_data_drift_alone_is_to_investigate_not_retrain():
    one = drift.build_alerts({"a": 0.5}, None, None, None)
    d1 = drift.retrain_decision(one)
    assert not d1["recommended"] and not d1["investigate"]
    two = drift.build_alerts({"a": 0.5, "b": 0.5}, None, None, None)
    d2 = drift.retrain_decision(two)
    assert not d2["recommended"]
    assert d2["investigate"] and "non dégradée" in (
        d2["investigate_reasons"][0]
    )
    assert d2["automatic"] is False


def test_investigate_message_without_labels():
    two = drift.build_alerts({"a": 0.5, "b": 0.5}, None, None, None)
    d = drift.retrain_decision(two, labels_available=False)
    assert "étiquettes non disponibles" in d["investigate_reasons"][0]


def test_perf_drop_plus_drift_is_recommended_not_investigate():
    alerts = drift.build_alerts(
        {"a": 0.5, "b": 0.5},
        None,
        {"f1_macro": 0.3, "roc_auc_ovr_macro": 0.675},
        BASE,
    )
    d = drift.retrain_decision(alerts)
    assert d["recommended"] and not d["investigate"]


# ----------------------------------------------------------- simulation
def test_simulate_data_drift_scales_only_given_columns():
    X, _ = make_data(200)
    out = sim.simulate_data_drift(X, {"avg_monthly_spend_ngn": 2.0})
    assert np.allclose(
        out["avg_monthly_spend_ngn"], 2 * X["avg_monthly_spend_ngn"]
    )
    pd.testing.assert_series_equal(
        out["late_payment_count"], X["late_payment_count"]
    )
    assert X["avg_monthly_spend_ngn"].max() < 60000  # original intact


def test_simulate_data_drift_missing_column():
    X, _ = make_data(50)
    with pytest.raises(KeyError):
        sim.simulate_data_drift(X, {"inconnue": 2.0})


def test_simulate_concept_drift():
    y = np.array([0, 1, 2] * 1000)
    assert np.array_equal(sim.simulate_concept_drift(y, 0.0), y)
    half = sim.simulate_concept_drift(y, 0.5, seed=1)
    assert 0.3 < (half != y).mean() < 0.5
    again = sim.simulate_concept_drift(y, 0.5, seed=1)
    assert np.array_equal(half, again)
    assert np.array_equal(y, np.array([0, 1, 2] * 1000))  # original intact
    with pytest.raises(ValueError):
        sim.simulate_concept_drift(y, 1.5)


# ------------------------------------------------------- bout en bout
def _batches(X_test, y_test):
    return [
        {"name": "sain", "simulated": False, "X": X_test, "y": y_test},
        {
            "name": "derive_donnees",
            "simulated": True,
            "X": sim.simulate_data_drift(X_test),
            "y": y_test,
        },
        {
            "name": "derive_concept",
            "simulated": True,
            "X": X_test,
            "y": sim.simulate_concept_drift(y_test, 0.8),
        },
    ]


@pytest.fixture(scope="module")
def report(fitted, tmp_path_factory):
    model, X_ref, X_test, y_test = fitted
    base = drift.performance_metrics(
        y_test, model.predict(X_test), model.predict_proba(X_test)
    )
    out = tmp_path_factory.mktemp("monitoring")
    rep = drift.run_monitoring(
        X_ref, _batches(X_test, y_test), model, base, out_dir=out,
        with_evidently=True, ref_sample=3000,
        model_info={"name": "m", "version": "1", "run_id": "r"},
    )
    return rep, out


def _batch(rep, name):
    return next(b for b in rep["batches"] if b["name"] == name)


def test_healthy_batch_has_no_alert(report):
    rep, _ = report
    b = _batch(rep, "sain")
    assert b["status"] == "ok" and b["alerts"] == []
    assert not b["retrain"]["recommended"]
    assert b["simulated"] is False


def test_data_drift_batch_flags_shifted_columns(report):
    rep, _ = report
    b = _batch(rep, "derive_donnees")
    names = {a["name"] for a in b["alerts"] if a["kind"] == "data_drift"}
    assert {"avg_monthly_spend_ngn",
            "recharge_frequency_per_month"} <= names
    assert "late_payment_count" not in names
    assert b["status"] == "critical" and b["simulated"] is True
    # Dans ces données fabriquées, la fréquence de recharge pèse sur la
    # cible : la dérive dégrade aussi la performance, donc réentraînement
    # recommandé. (Dérive sans baisse de performance : tests unitaires.)
    assert b["retrain"]["recommended"]
    assert not b["retrain"]["investigate"]


def test_concept_drift_hits_performance_not_features(report):
    rep, _ = report
    b = _batch(rep, "derive_concept")
    kinds = {a["kind"] for a in b["alerts"] if a["level"] == "critical"}
    assert "performance" in kinds
    assert "data_drift" not in {a["kind"] for a in b["alerts"]}
    assert b["retrain"]["recommended"]


def test_evidently_report_and_per_column(report):
    rep, out = report
    ev = _batch(rep, "derive_donnees")["evidently"]
    assert Path(ev["html"]).exists()
    assert ev["per_column"]["avg_monthly_spend_ngn"]["drifted"] is True
    assert ev["per_column"]["late_payment_count"]["drifted"] is False
    assert ev["drifted_count"] == sum(
        v["drifted"] for v in ev["per_column"].values()
    )


def test_report_json_is_valid_and_complete(report):
    rep, out = report
    text = (out / drift.REPORT_NAME).read_text(encoding="utf-8")
    assert "NaN" not in text
    data = json.loads(text)
    assert data["reference"]["n_rows_used"] == 3000
    assert [b["name"] for b in data["batches"]] == [
        "sain", "derive_donnees", "derive_concept"
    ]


def test_without_evidently_no_html(fitted, tmp_path):
    model, X_ref, X_test, y_test = fitted
    rep = drift.run_monitoring(
        X_ref, _batches(X_test, y_test)[:1], model, None,
        out_dir=tmp_path, with_evidently=False,
    )
    assert rep["batches"][0]["evidently"] is None
    assert not list(tmp_path.glob("*.html"))


def test_evidently_failure_is_recorded_not_hidden(
    fitted, tmp_path, monkeypatch
):
    model, X_ref, X_test, y_test = fitted

    def boom(*args, **kwargs):
        raise RuntimeError("panne simulée")

    monkeypatch.setattr(drift, "evidently_drift", boom)
    rep = drift.run_monitoring(
        X_ref, _batches(X_test, y_test)[:1], model, None,
        out_dir=tmp_path, with_evidently=True, ref_sample=1000,
    )
    assert "panne simulée" in rep["batches"][0]["evidently"]["error"]


# ----------------------------------------------------------- dashboard
def _app(monkeypatch, path):
    # Streamlit n'est pas dans requirements.txt : sans lui, la CI saute
    # ces deux tests au lieu d'échouer.
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MONITORING_REPORT", str(path))
    return AppTest.from_file(str(DASHBOARD), default_timeout=60).run()


def test_dashboard_renders_and_switches_batch(report, monkeypatch):
    rep, out = report
    at = _app(monkeypatch, out / drift.REPORT_NAME)
    assert not at.exception
    assert at.title[0].value.startswith("Monitoring")
    assert any("simulé" in w.value for w in at.warning)
    assert at.success  # lot « sain » : statut OK
    spec = at.get("vega_lite_chart")[0].proto.spec
    chart = json.loads(spec)
    scale = chart["layer"][0]["encoding"]["x"]["scale"]
    assert scale["domain"][0] == 0
    assert scale["domain"][1] >= 0.25  # l'axe atteint au moins le seuil
    assert len(chart["layer"]) == 3  # barres, valeurs, seuils
    at.sidebar.selectbox[0].select("derive_donnees").run()
    assert not at.exception
    assert any("CRITIQUE" in e.value for e in at.error)


def test_dashboard_without_report(tmp_path, monkeypatch):
    at = _app(monkeypatch, tmp_path / "absent.json")
    assert not at.exception
    assert any("introuvable" in e.value for e in at.error)


# --------------------------------------------- compatibilité pandas 3.1
def test_compat_shim_is_noop_when_function_exists(monkeypatch):
    import pandas.core.dtypes.common as common

    sentinel = object()
    monkeypatch.setattr(
        common, "infer_dtype_from_object", sentinel, raising=False
    )
    assert drift.ensure_evidently_compat() is False
    assert common.infer_dtype_from_object is sentinel


def test_compat_shim_restores_missing_function(monkeypatch):
    import pandas.core.dtypes.common as common

    monkeypatch.delattr(
        common, "infer_dtype_from_object", raising=False
    )
    assert drift.ensure_evidently_compat() is True
    fn = common.infer_dtype_from_object
    assert fn(np.number) is np.number
    assert fn(np.dtype("float64")) is np.float64
    assert fn("int64") is np.int64
    assert fn(object()) is not None  # type inconnu : renvoyé tel quel
    monkeypatch.undo()


def test_evidently_imports_after_shim():
    drift.ensure_evidently_compat()
    from evidently import Report  # noqa: F401


def test_psi_chart_axis_is_fixed_and_labels_not_truncated():
    pytest.importorskip("streamlit")
    from src.monitoring.dashboard import psi_chart

    table = pd.DataFrame(
        {"Variable": ["a", "b"], "PSI": [0.0001, 0.0002]}
    )
    spec = psi_chart(
        table, {"psi_warning": 0.10, "psi_critical": 0.25}
    ).to_dict()
    bars = spec["layer"][0]["encoding"]
    # PSI minuscules : l'axe reste calé sur le seuil critique (>= 0,25)
    assert bars["x"]["scale"]["domain"][1] >= 0.25
    assert bars["y"]["axis"]["labelLimit"] >= 300
    # valeurs lisibles sur thème sombre : couleur explicite, pas le noir
    text_mark = spec["layer"][1]["mark"]
    assert text_mark["type"] == "text" and text_mark["color"] != "black"
    big = psi_chart(
        pd.DataFrame({"Variable": ["a"], "PSI": [2.6]}),
        {"psi_warning": 0.10, "psi_critical": 0.25},
    ).to_dict()
    assert big["layer"][0]["encoding"]["x"]["scale"]["domain"][1] > 2.6

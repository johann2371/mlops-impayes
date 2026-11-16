"""Moteur de monitoring (tâche 10).

Pour chaque lot de données (« batch ») il calcule :
  - la dérive des données : PSI par variable + rapport Evidently ;
  - la dérive des prédictions : PSI de la répartition des classes prédites ;
  - la performance (F1 macro, AUC un-contre-tous) quand les étiquettes
    sont connues, comparée à celle du modèle en production (registre) ;
  - des alertes (warning / critical). Le réentraînement n'est recommandé
    que si la performance baisse de façon critique ; une dérive des
    données seule est « à investiguer ». Rien n'est jamais automatique.

Le résultat est écrit dans ``reports/monitoring/monitoring_report.json``,
lu par ``src/monitoring/dashboard.py``.

Usage :
  python -m src.monitoring.drift
  python -m src.monitoring.drift --no-evidently   # sans rapport HTML
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Seuils d'alerte : choix de conception, à aligner avec le rapport.
THRESHOLDS = {
    "psi_warning": 0.10,
    "psi_critical": 0.25,
    "drop_warning": 0.03,   # baisse absolue de F1 macro / AUC (points)
    "drop_critical": 0.05,
    "min_critical_features": 2,
}
EPS = 1e-4
MAX_DISCRETE = 12
LABELS = [0, 1, 2]
REPORT_NAME = "monitoring_report.json"
RETRAIN_NOTE = (
    "Recommandation seulement : un nouveau modèle doit être entraîné, "
    "comparé au modèle en production dans le registre, puis validé. "
    "Aucun remplacement automatique."
)


# --------------------------------------------------------------- PSI
def _psi_from_props(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    p_ref = np.clip(p_ref, EPS, None)
    p_cur = np.clip(p_cur, EPS, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index entre deux séries.

    Variable continue : tranches définies par les quantiles de la
    référence. Variable discrète (<= 12 valeurs) ou catégorielle :
    une tranche par modalité.
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.empty or cur.empty:
        raise ValueError("Série vide : PSI incalculable")
    numeric = pd.api.types.is_numeric_dtype(
        ref
    ) and not pd.api.types.is_bool_dtype(ref)
    if numeric and ref.nunique() > MAX_DISCRETE:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            p_ref = np.histogram(ref, edges)[0] / len(ref)
            p_cur = np.histogram(cur, edges)[0] / len(cur)
            return _psi_from_props(p_ref, p_cur)
    cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    p_ref = ref.value_counts(normalize=True).reindex(cats, fill_value=0)
    p_cur = cur.value_counts(normalize=True).reindex(cats, fill_value=0)
    return _psi_from_props(p_ref.to_numpy(), p_cur.to_numpy())


def data_drift_psi(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """PSI de chaque colonne commune à ``reference`` et ``current``."""
    return {c: psi(reference[c], current[c]) for c in reference.columns}


def prediction_psi(ref_pred, cur_pred) -> float:
    """PSI de la répartition des classes prédites."""
    ref = np.bincount(np.asarray(ref_pred, dtype=int), minlength=3)
    cur = np.bincount(np.asarray(cur_pred, dtype=int), minlength=3)
    return _psi_from_props(ref / ref.sum(), cur / cur.sum())


# ----------------------------------------------------------- Evidently
def _infer_dtype_from_object(dtype):
    """Équivalent simplifié de la fonction retirée de pandas 3.1.0."""
    from pandas.api.extensions import ExtensionDtype

    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return dtype
    if isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype.type
    try:
        return pd.api.types.pandas_dtype(dtype).type
    except TypeError:
        return dtype


def ensure_evidently_compat() -> bool:
    """Rend Evidently 0.7.23 importable avec pandas 3.1.0.

    Evidently importe ``infer_dtype_from_object`` depuis
    ``pandas.core.dtypes.common``, fonction supprimée de pandas 3.1.0.
    Il ne l'emploie que dans une classe de tests (``TestColumnsType``)
    que ce projet n'utilise pas. Sans effet si la fonction existe déjà
    (pandas <= 3.0) ou si une version ultérieure corrige le problème.
    Renvoie True si la rustine a été appliquée.
    """
    import pandas.core.dtypes.common as common

    if hasattr(common, "infer_dtype_from_object"):
        return False
    common.infer_dtype_from_object = _infer_dtype_from_object
    return True


def evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: str | Path | None = None,
) -> dict:
    """Dérive par variable selon Evidently (DataDriftPreset)."""
    ensure_evidently_compat()
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    cols = list(reference.columns)
    num = [
        c
        for c in cols
        if pd.api.types.is_numeric_dtype(reference[c])
        and not pd.api.types.is_bool_dtype(reference[c])
    ]
    cat = [c for c in cols if c not in num]
    definition = DataDefinition(
        numerical_columns=num, categorical_columns=cat
    )
    d_ref = Dataset.from_pandas(reference, data_definition=definition)
    d_cur = Dataset.from_pandas(current, data_definition=definition)
    snapshot = Report([DataDriftPreset()]).run(d_cur, d_ref)
    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))

    per_column, share, count = {}, None, None
    for item in snapshot.dict()["metrics"]:
        cfg = item.get("config", {})
        kind = str(cfg.get("type", ""))
        if kind.endswith("DriftedColumnsCount"):
            share = float(item["value"]["share"])
            count = int(item["value"]["count"])
        elif kind.endswith("ValueDrift"):
            dist, thr = float(item["value"]), float(cfg["threshold"])
            per_column[cfg["column"]] = {
                "method": cfg["method"],
                "distance": dist,
                "threshold": thr,
                "drifted": dist > thr,
            }
    return {
        "drifted_share": share,
        "drifted_count": count,
        "per_column": per_column,
        "html": str(html_path) if html_path is not None else None,
    }


# --------------------------------------------------------- performance
def performance_metrics(y_true, y_pred, y_proba) -> dict:
    """F1 macro, exactitude et AUC un-contre-tous (None si impossible)."""
    y_true = np.asarray(y_true)
    kw = {"labels": LABELS, "average": "macro", "zero_division": 0}
    auc = None
    # L'AUC un-contre-tous exige les trois classes dans y_true ; selon la
    # version de scikit-learn, son absence lève une erreur ou renvoie NaN.
    if set(LABELS) <= set(np.unique(y_true).tolist()):
        auc = float(
            roc_auc_score(
                y_true,
                np.asarray(y_proba),
                multi_class="ovr",
                average="macro",
                labels=LABELS,
            )
        )
        if np.isnan(auc):
            auc = None
    return {
        "f1_macro": float(f1_score(y_true, y_pred, **kw)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc_ovr_macro": auc,
    }


# -------------------------------------------------------------- alertes
def _level(value: float, warning: float, critical: float) -> str | None:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return None


def _alert(level, kind, name, value, threshold, message) -> dict:
    return {
        "level": level,
        "kind": kind,
        "name": name,
        "value": round(float(value), 4),
        "threshold": threshold,
        "message": message,
    }


def build_alerts(
    psi_by_column: dict,
    pred_psi: float | None,
    metrics: dict | None,
    baseline: dict | None,
    thresholds: dict | None = None,
) -> list[dict]:
    """Liste des alertes d'un lot, selon les seuils."""
    t = {**THRESHOLDS, **(thresholds or {})}
    alerts: list[dict] = []
    for col, value in psi_by_column.items():
        lvl = _level(value, t["psi_warning"], t["psi_critical"])
        if lvl:
            limit = t["psi_critical" if lvl == "critical" else "psi_warning"]
            alerts.append(
                _alert(lvl, "data_drift", col, value, limit,
                       f"PSI de {col} = {value:.3f}")
            )
    if pred_psi is not None:
        lvl = _level(pred_psi, t["psi_warning"], t["psi_critical"])
        if lvl:
            limit = t["psi_critical" if lvl == "critical" else "psi_warning"]
            alerts.append(
                _alert(lvl, "prediction_drift", "classes_predites",
                       pred_psi, limit,
                       f"PSI des classes prédites = {pred_psi:.3f}")
            )
    if metrics and baseline:
        for key in ("f1_macro", "roc_auc_ovr_macro"):
            cur, ref = metrics.get(key), baseline.get(key)
            if cur is None or ref is None:
                continue
            drop = round(ref - cur, 6)  # évite les écarts de flottants
            lvl = _level(drop, t["drop_warning"], t["drop_critical"])
            if lvl:
                limit = t["drop_critical" if lvl == "critical"
                          else "drop_warning"]
                alerts.append(
                    _alert(lvl, "performance", key, drop, limit,
                           f"{key} : {cur:.4f} contre {ref:.4f} en "
                           f"production (baisse de {drop:.4f})")
                )
    return alerts


def batch_status(alerts: list[dict]) -> str:
    levels = {a["level"] for a in alerts}
    if "critical" in levels:
        return "critical"
    if "warning" in levels:
        return "warning"
    return "ok"


def retrain_decision(
    alerts: list[dict],
    thresholds=None,
    labels_available: bool = True,
) -> dict:
    """Décide s'il faut recommander un réentraînement.

    Règle (cf. question R10.3 de l'énoncé : une dérive des données ne
    justifie pas à elle seule de remplacer le modèle) :
      - réentraînement **recommandé** seulement si la performance baisse
        de façon critique (F1 macro ou AUC) ;
      - dérive critique d'au moins ``min_critical_features`` variables
        sans baisse de performance : **à investiguer**, pas de
        recommandation.
    Dans tous les cas, rien n'est déclenché automatiquement.
    """
    t = {**THRESHOLDS, **(thresholds or {})}
    critical = [a for a in alerts if a["level"] == "critical"]
    perf = [a for a in critical if a["kind"] == "performance"]
    feats = [a for a in critical if a["kind"] == "data_drift"]
    reasons = []
    if perf:
        reasons.append(
            "performance : " + ", ".join(a["name"] for a in perf)
        )
    investigate_reasons = []
    if len(feats) >= t["min_critical_features"] and not perf:
        text = (
            f"{len(feats)} variables en dérive critique : "
            + ", ".join(a["name"] for a in feats)
        )
        if labels_available:
            text += " (performance non dégradée)"
        else:
            text += (
                " ; étiquettes non disponibles : mesurer la performance "
                "dès qu'elles arrivent"
            )
        investigate_reasons.append(text)
    return {
        "recommended": bool(reasons),
        "reasons": reasons,
        "investigate": bool(investigate_reasons),
        "investigate_reasons": investigate_reasons,
        "automatic": False,
        "note": RETRAIN_NOTE,
    }


# --------------------------------------------------------- orchestration
def evaluate_batch(
    batch: dict,
    model,
    reference: pd.DataFrame,
    reference_pred,
    baseline: dict | None,
    html_dir: str | Path | None = None,
    thresholds: dict | None = None,
) -> dict:
    """Calcule dérives, performance et alertes pour un lot."""
    X, y = batch["X"], batch.get("y")
    pred = np.asarray(model.predict(X))
    proba = model.predict_proba(X)
    psi_cols = data_drift_psi(reference, X)
    pred_psi = prediction_psi(reference_pred, pred)
    metrics = None if y is None else performance_metrics(y, pred, proba)

    evidently = None
    if html_dir is not None:
        try:
            evidently = evidently_drift(
                reference, X, Path(html_dir) / f"{batch['name']}.html"
            )
        except Exception as exc:  # noqa: BLE001
            evidently = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[!] Evidently a échoué ({batch['name']}) : {exc}")

    alerts = build_alerts(psi_cols, pred_psi, metrics, baseline, thresholds)
    return {
        "name": batch["name"],
        "description": batch.get("description", ""),
        "simulated": bool(batch.get("simulated", False)),
        "n_rows": int(len(X)),
        "metrics": metrics,
        "psi": psi_cols,
        "prediction_psi": pred_psi,
        "evidently": evidently,
        "alerts": alerts,
        "status": batch_status(alerts),
        "retrain": retrain_decision(
            alerts, thresholds, labels_available=y is not None
        ),
    }


def _clean(obj):
    """Rend l'objet sérialisable en JSON (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_monitoring(
    reference_X: pd.DataFrame,
    batches: list[dict],
    model,
    baseline: dict | None,
    out_dir: str | Path = "reports/monitoring",
    with_evidently: bool = True,
    ref_sample: int = 100_000,
    seed: int = 42,
    model_info: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    """Évalue tous les lots et écrit le rapport JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_X
    if ref_sample and len(reference) > ref_sample:
        reference = reference.sample(ref_sample, random_state=seed)
    reference_pred = np.asarray(model.predict(reference))

    html_dir = out_dir if with_evidently else None
    results = [
        evaluate_batch(b, model, reference, reference_pred, baseline,
                       html_dir, thresholds)
        for b in batches
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_info or {},
        "baseline": baseline,
        "thresholds": {**THRESHOLDS, **(thresholds or {})},
        "reference": {
            "n_rows_total": int(len(reference_X)),
            "n_rows_used": int(len(reference)),
        },
        "batches": results,
    }
    report = _clean(report)
    path = out_dir / REPORT_NAME
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    from src.features import build_features as bf
    from src.models import registry, train
    from src.monitoring import simulate_drift as sim

    import mlflow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out-dir", default="reports/monitoring")
    parser.add_argument("--ref-sample", type=int, default=100_000)
    parser.add_argument("--concept-fraction", type=float, default=0.5)
    parser.add_argument("--no-evidently", action="store_true")
    args = parser.parse_args()

    params = train.load_params(args.params)
    cfg = params["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])

    # Même découpage que l'entraînement (lu dans les métadonnées).
    meta = json.loads(
        Path(params["outputs"]["model_meta_path"]).read_text(
            encoding="utf-8"
        )
    )
    params["data"]["sample_frac"] = meta["sample_frac"]
    params["seed"] = meta["seed"]
    df = train.load_dataset(params)
    X_train, X_test, _, y_test = bf.split_data(
        df, meta["feature_set"], meta["test_size"], meta["seed"]
    )

    client = mlflow.MlflowClient()
    prod = registry.get_by_alias(
        client, registry.MODEL_NAME, registry.PRODUCTION
    )
    if prod is None:
        raise SystemExit(
            "Aucun modèle en production : lancer "
            "`python -m src.models.registry` d'abord."
        )
    run_metrics = client.get_run(prod.run_id).data.metrics
    baseline = {
        "f1_macro": run_metrics.get("test_f1_macro"),
        "roc_auc_ovr_macro": run_metrics.get("test_roc_auc_ovr_macro"),
    }
    model = registry.load_production_model()

    batches = [
        {
            "name": "sain",
            "description": "Jeu de test réel : aucune dérive attendue.",
            "simulated": False,
            "X": X_test,
            "y": y_test,
        },
        {
            "name": "derive_donnees",
            "description": (
                "SIMULÉ à partir du jeu de test : dépenses moyennes "
                "x1,5 et fréquence de recharge x0,6."
            ),
            "simulated": True,
            "X": sim.simulate_data_drift(X_test),
            "y": y_test,
        },
        {
            "name": "derive_concept",
            "description": (
                f"SIMULÉ : {args.concept_fraction:.0%} des étiquettes "
                "tirées au hasard (variables inchangées)."
            ),
            "simulated": True,
            "X": X_test,
            "y": sim.simulate_concept_drift(
                y_test, args.concept_fraction, meta["seed"]
            ),
        },
    ]
    report = run_monitoring(
        X_train,
        batches,
        model,
        baseline,
        out_dir=args.out_dir,
        with_evidently=not args.no_evidently,
        ref_sample=args.ref_sample,
        seed=meta["seed"],
        model_info={
            "name": registry.MODEL_NAME,
            "version": str(prod.version),
            "run_id": prod.run_id,
        },
    )
    for b in report["batches"]:
        r = b["retrain"]
        retrain = (
            "recommandé" if r["recommended"]
            else "à investiguer" if r["investigate"] else "non"
        )
        print(
            f"{b['name']:<16} statut={b['status']:<9} "
            f"alertes={len(b['alerts']):<2} réentraînement={retrain}"
        )
    print(f"Rapport : {Path(args.out_dir) / REPORT_NAME}")


if __name__ == "__main__":
    main()

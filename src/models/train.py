"""Entraînement et suivi MLflow (tâche 5).

Deux modes :
- sans option : un run « final » avec les choix de ``params.yaml``,
  évalué sur le jeu de **test** ; sauvegarde du modèle et des métriques ;
- ``--ablation`` : compare plusieurs configurations sur un jeu de
  **validation** prélevé dans le train. Le jeu de test n'est pas utilisé
  pour choisir, sinon la métrique finale serait optimiste.

Lancer depuis la racine du dépôt : ``python -m src.models.train``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import build_features as bf
from src.models import evaluate as ev

MODELS = ("hist_gradient_boosting", "logistic_regression")


def load_params(path: str | Path = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(name: str, class_weight, seed: int, hyper: dict):
    """Classifieur scikit-learn. ``class_weight`` : None ou 'balanced'."""
    hyper = dict(hyper or {})
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight=class_weight, random_state=seed, **hyper
        )
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight=class_weight, random_state=seed, **hyper
        )
    raise ValueError(f"Modèle inconnu : {name!r}. Choix : {list(MODELS)}")


def load_dataset(params: dict) -> pd.DataFrame:
    """Charge le CSV brut ; sous-échantillon optionnel (essais rapides)."""
    df = bf.load_raw(params["data"]["raw_path"])
    frac = params["data"].get("sample_frac")
    if frac:
        df = df.sample(frac=frac, random_state=params["seed"])
    return df


def _setup_mlflow(cfg: dict) -> None:
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    name = cfg["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(
            name, artifact_location=cfg.get("artifact_location")
        )
    mlflow.set_experiment(name)


# mlflow 3.x sérialise avec skops et refuse les types « non fiables ».
# On déclare ici, explicitement, les seuls types propres au projet.
TRUSTED_TYPES = [
    "numpy.dtype",
    f"{bf.__name__}._identity",
    f"{bf.__name__}._to_float",
    f"{bf.__name__}.add_ratio_features",
    # Arbres de HistGradientBoosting : modèle entraîné par nous.
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def _log_model(model) -> None:
    """Compatible avec les deux signatures de mlflow (name / artifact_path)."""
    kwargs = {"skops_trusted_types": TRUSTED_TYPES}
    try:
        mlflow.sklearn.log_model(model, name="model", **kwargs)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)


def run_experiment(
    params: dict,
    df: pd.DataFrame,
    evaluate_on: str = "test",
    save_outputs: bool = True,
) -> dict:
    """Entraîne, évalue et journalise un run MLflow.

    ``evaluate_on`` : 'test' (run final) ou 'val' (comparaison).
    Retourne les métriques préfixées (``test_f1_macro``, ...)."""
    if evaluate_on not in ("test", "val"):
        raise ValueError("evaluate_on doit valoir 'test' ou 'val'")
    seed = params["seed"]
    fs = params["features"]["feature_set"]
    ratios = bool(params["features"]["add_ratios"])
    mname = params["model"]["name"]
    cw = params["model"].get("class_weight")

    X_train, X_test, y_train, y_test = bf.split_data(
        df, fs, params["data"]["test_size"], seed
    )
    if evaluate_on == "val":
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X_train,
            y_train,
            test_size=params["data"]["validation_size"],
            random_state=seed,
            stratify=y_train,
        )
    else:
        X_fit, y_fit, X_eval, y_eval = X_train, y_train, X_test, y_test

    pipeline = Pipeline(
        [
            ("features", bf.build_pipeline(fs, ratios)),
            (
                "model",
                build_model(mname, cw, seed, params["model"].get(mname)),
            ),
        ]
    )
    t0 = time.time()
    pipeline.fit(X_fit, y_fit)
    fit_seconds = time.time() - t0

    y_pred = pipeline.predict(X_eval)
    y_proba = pipeline.predict_proba(X_eval)
    metrics = ev.compute_metrics(y_eval, y_pred, y_proba)
    metrics["baseline_f1_macro"] = ev.baseline_f1_macro(y_fit, y_eval)
    cm = ev.confusion(y_eval, y_pred)

    run_name = f"{mname}-{fs}-ratios{int(ratios)}-cw{cw}-{evaluate_on}"
    _setup_mlflow(params["mlflow"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": mname,
                "feature_set": fs,
                "add_ratios": ratios,
                "class_weight": cw,
                "seed": seed,
                "test_size": params["data"]["test_size"],
                "validation_size": params["data"]["validation_size"],
                "sample_frac": params["data"].get("sample_frac"),
                "n_fit": len(X_fit),
                "n_eval": len(X_eval),
                "evaluated_on": evaluate_on,
                **{
                    f"hp_{k}": v
                    for k, v in (params["model"].get(mname) or {}).items()
                },
            }
        )
        mlflow.set_tag(
            "stage", "final" if evaluate_on == "test" else "ablation"
        )
        mlflow.log_metrics(
            {f"{evaluate_on}_{k}": v for k, v in metrics.items()}
        )
        mlflow.log_metric("fit_seconds", fit_seconds)
        with tempfile.TemporaryDirectory() as tmp:
            png = ev.save_confusion_matrix(
                cm, Path(tmp) / "confusion_matrix.png"
            )
            mlflow.log_artifact(str(png))
        _log_model(pipeline)

    result = {f"{evaluate_on}_{k}": v for k, v in metrics.items()}
    result["confusion_matrix"] = cm.tolist()
    if save_outputs and evaluate_on == "test":
        _save_outputs(params, pipeline, X_train, result, fit_seconds)
    return result


def _save_outputs(params, pipeline, X_train, result, fit_seconds) -> None:
    out = params["outputs"]
    for key in ("model_path", "model_meta_path", "metrics_path"):
        Path(out[key]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out["model_path"])
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_set": params["features"]["feature_set"],
        "add_ratios": bool(params["features"]["add_ratios"]),
        "model": params["model"]["name"],
        "class_weight": params["model"].get("class_weight"),
        "seed": params["seed"],
        "test_size": params["data"]["test_size"],
        "sample_frac": params["data"].get("sample_frac"),
        "classes": bf.TARGET_ORDER,
        "input_columns": list(X_train.columns),
        "fit_seconds": round(fit_seconds, 1),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    Path(out["model_meta_path"]).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(out["metrics_path"]).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ev.save_confusion_matrix(
        pd.DataFrame(result["confusion_matrix"]).to_numpy(),
        out["confusion_matrix_path"],
    )


def run_ablation(params: dict, df: pd.DataFrame, models=None) -> pd.DataFrame:
    """Compare jeux de variables x ratios x pondération, sur validation."""
    models = models or [params["model"]["name"]]
    rows = []
    grid = itertools.product(
        models, bf.FEATURE_SETS, (False, True), (None, "balanced")
    )
    for mname, fs, ratios, cw in grid:
        p = copy.deepcopy(params)
        p["model"]["name"] = mname
        p["model"]["class_weight"] = cw
        p["features"]["feature_set"] = fs
        p["features"]["add_ratios"] = ratios
        res = run_experiment(p, df, evaluate_on="val", save_outputs=False)
        rows.append(
            {
                "model": mname,
                "feature_set": fs,
                "add_ratios": ratios,
                "class_weight": cw,
                **{
                    k: round(v, 4)
                    for k, v in res.items()
                    if k
                    in (
                        "val_f1_macro",
                        "val_precision_macro",
                        "val_recall_macro",
                        "val_roc_auc_ovr_macro",
                        "val_baseline_f1_macro",
                    )
                },
            }
        )
    table = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    path = Path(params["outputs"]["ablation_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", help="CSV brut (remplace params.yaml)")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--feature-set", choices=list(bf.FEATURE_SETS))
    parser.add_argument(
        "--add-ratios", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--class-weight", choices=["none", "balanced"])
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="compare les configurations sur validation (pas de test)",
    )
    parser.add_argument(
        "--ablation-models",
        nargs="+",
        choices=MODELS,
        help="modèles à comparer en mode --ablation",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    if args.data:
        params["data"]["raw_path"] = args.data
    if args.sample_frac:
        params["data"]["sample_frac"] = args.sample_frac
    if args.model:
        params["model"]["name"] = args.model
    if args.feature_set:
        params["features"]["feature_set"] = args.feature_set
    if args.add_ratios is not None:
        params["features"]["add_ratios"] = args.add_ratios
    if args.class_weight:
        params["model"]["class_weight"] = (
            None if args.class_weight == "none" else args.class_weight
        )

    df = load_dataset(params)
    print(f"Données : {df.shape} depuis {params['data']['raw_path']}")
    if args.ablation:
        table = run_ablation(params, df, args.ablation_models)
        print(table.to_string(index=False))
        print(f"Tableau enregistré : {params['outputs']['ablation_path']}")
        return
    res = run_experiment(params, df, evaluate_on="test")
    keys = (
        "test_f1_macro",
        "test_precision_macro",
        "test_recall_macro",
        "test_roc_auc_ovr_macro",
        "test_baseline_f1_macro",
    )
    for k in keys:
        print(f"{k} : {res[k]:.4f}")
    print("Matrice de confusion (réel x prédit) :", res["confusion_matrix"])
    print(f"Modèle : {params['outputs']['model_path']}")


if __name__ == "__main__":
    main()

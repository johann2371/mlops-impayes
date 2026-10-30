"""Évaluation multiclasse (tâche 5).

La cible ``default_risk`` a trois classes (low, medium, high) : les
métriques sont donc calculées en moyenne *macro* (chaque classe pèse
autant) et l'AUC est de type un-contre-tous (OvR). Voir docs/features.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import build_features as bf  # noqa: E402

CLASS_NAMES = bf.TARGET_ORDER
LABELS = list(range(len(CLASS_NAMES)))


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Précision, rappel, F1 (macro et par classe), exactitude, AUC OvR.

    ``y_proba`` : probabilités, une colonne par classe (ordre 0, 1, 2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    kw = {"labels": LABELS, "zero_division": 0}
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", **kw)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", **kw)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=LABELS,
            )
        ),
    }
    per_class = {
        "precision": precision_score(y_true, y_pred, average=None, **kw),
        "recall": recall_score(y_true, y_pred, average=None, **kw),
        "f1": f1_score(y_true, y_pred, average=None, **kw),
    }
    for i, name in enumerate(CLASS_NAMES):
        for key, values in per_class.items():
            metrics[f"{key}_{name}"] = float(values[i])
        metrics[f"roc_auc_{name}"] = float(
            roc_auc_score((y_true == i).astype(int), y_proba[:, i])
        )
    return metrics


def baseline_f1_macro(y_fit, y_eval) -> float:
    """F1 macro d'un modèle qui prédit toujours la classe majoritaire.

    Sert de plancher : un modèle utile doit faire nettement mieux."""
    majority = int(np.bincount(np.asarray(y_fit)).argmax())
    constant = np.full(len(y_eval), majority)
    return float(
        f1_score(
            y_eval, constant, labels=LABELS, average="macro", zero_division=0
        )
    )


def confusion(y_true, y_pred) -> np.ndarray:
    """Matrice de confusion 3x3 (lignes = réel, colonnes = prédit)."""
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def save_confusion_matrix(cm: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(LABELS)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(LABELS)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title("Matrice de confusion")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in LABELS:
        for j in LABELS:
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """Réévalue un modèle sauvegardé sur le jeu de test (même split)."""
    from src.models import train

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model", default=None, help="fichier .joblib")
    parser.add_argument("--data", default=None, help="CSV brut")
    args = parser.parse_args()

    params = train.load_params(args.params)
    if args.data:
        params["data"]["raw_path"] = args.data
    model_path = args.model or params["outputs"]["model_path"]
    model = joblib.load(model_path)

    # Même split que l'entraînement : lu dans les métadonnées du modèle,
    # sinon des lignes d'entraînement pourraient se retrouver dans le test.
    meta = json.loads(
        Path(params["outputs"]["model_meta_path"]).read_text(encoding="utf-8")
    )
    params["data"]["sample_frac"] = meta["sample_frac"]
    params["data"]["test_size"] = meta["test_size"]
    params["seed"] = meta["seed"]
    fs = meta["feature_set"]

    df = train.load_dataset(params)
    _, X_test, _, y_test = bf.split_data(
        df, fs, params["data"]["test_size"], params["seed"]
    )
    metrics = compute_metrics(
        y_test, model.predict(X_test), model.predict_proba(X_test)
    )
    print(json.dumps(metrics, indent=2))
    print("Matrice de confusion (réel x prédit) :")
    print(confusion(y_test, model.predict(X_test)))


if __name__ == "__main__":
    main()

"""Registre de modèles MLflow (tâche 6).

Rôle : inscrire dans le registre le modèle du run final (jeu complet),
lui donner le statut ``staging``, puis le promouvoir en ``production``
seulement s'il n'est pas moins bon que le modèle déjà en production.

Les statuts sont des alias MLflow (les « stages » sont dépréciés) :
  staging     candidat inscrit, en attente de validation
  production  modèle servi par l'API
  previous    ancien modèle de production, conservé pour le rollback

Usage :
  python -m src.models.registry              # inscrit, compare, promeut
  python -m src.models.registry --dry-run    # inscrit mais ne promeut pas
  python -m src.models.registry --show       # affiche l'état du registre
  python -m src.models.registry --run-id ID  # candidat précis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "impayes-telecom-model"
PRIMARY_METRIC = "test_roc_auc_ovr_macro"
SECONDARY_METRIC = "test_f1_macro"
STAGING = "staging"
PRODUCTION = "production"
PREVIOUS = "previous"
TAGGED_PARAMS = (
    "model",
    "feature_set",
    "add_ratios",
    "class_weight",
    "seed",
    "n_fit",
    "n_eval",
)


def load_mlflow_config(path: str | Path = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["mlflow"]


def find_final_run(client: MlflowClient, experiment_name: str):
    """Dernier run ``stage=final`` entraîné sur le jeu complet.

    Les essais sur échantillon (``sample_frac`` renseigné) sont ignorés :
    MLflow enregistre ``sample_frac=None`` sous la forme du texte « None ».
    """
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Expérience introuvable : {experiment_name}")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'final'",
        order_by=["attributes.start_time DESC"],
    )
    full = [
        r
        for r in runs
        if r.data.params.get("sample_frac") in (None, "None")
    ]
    if not full:
        raise ValueError(
            "Aucun run final sur le jeu complet : lancer "
            "`python -m src.models.train` sans --sample-frac."
        )
    return full[0]


def register_candidate(
    client: MlflowClient, run, name: str = MODEL_NAME
) -> ModelVersion:
    """Inscrit le modèle du run (une seule version par run) en staging."""
    run_id = run.info.run_id
    existing = [
        v
        for v in client.search_model_versions(f"name='{name}'")
        if v.run_id == run_id
    ]
    if existing:
        version = existing[0]
    else:
        version = mlflow.register_model(f"runs:/{run_id}/model", name)
    v = version.version
    for key in TAGGED_PARAMS:
        if key in run.data.params:
            client.set_model_version_tag(name, v, key, run.data.params[key])
    for key in (PRIMARY_METRIC, SECONDARY_METRIC):
        if key in run.data.metrics:
            client.set_model_version_tag(
                name, v, key, f"{run.data.metrics[key]:.4f}"
            )
    client.set_model_version_tag(name, v, "run_id", run_id)
    client.update_model_version(
        name,
        v,
        description=(
            "Prédiction du risque d'impayé (3 classes). "
            f"Run MLflow {run_id}, évalué sur le jeu de test."
        ),
    )
    client.set_registered_model_alias(name, STAGING, v)
    return client.get_model_version(name, v)


def get_by_alias(client: MlflowClient, name: str, alias: str):
    """Version portant l'alias, ou ``None`` s'il n'existe pas."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def decide(
    candidate: dict,
    production: dict | None,
    min_delta: float = 0.0,
    metric: str = PRIMARY_METRIC,
) -> tuple[bool, str]:
    """Règle de promotion : l'AUC macro du candidat doit égaler ou
    dépasser celle de la production (+ ``min_delta``).

    L'AUC est la mesure principale : le F1 macro peut monter sans que le
    modèle sépare mieux les classes (effet de ``class_weight``).
    """
    if metric not in candidate:
        raise KeyError(f"Métrique absente du candidat : {metric}")
    if production is None:
        return True, "aucun modèle en production : première promotion"
    if metric not in production:
        raise KeyError(f"Métrique absente de la production : {metric}")
    cand, prod = candidate[metric], production[metric]
    if cand >= prod + min_delta:
        return True, f"{metric} {cand:.4f} >= {prod:.4f} + {min_delta}"
    return False, f"{metric} {cand:.4f} < {prod:.4f} + {min_delta}"


def promote(client: MlflowClient, name: str, version: str) -> None:
    """Passe la version en production et garde l'ancienne (rollback)."""
    old = get_by_alias(client, name, PRODUCTION)
    if old is not None and str(old.version) != str(version):
        client.set_registered_model_alias(name, PREVIOUS, str(old.version))
    client.set_registered_model_alias(name, PRODUCTION, str(version))


def rollback(client: MlflowClient, name: str = MODEL_NAME) -> str:
    """Remet en production la version marquée ``previous``."""
    prev = get_by_alias(client, name, PREVIOUS)
    if prev is None:
        raise ValueError("Aucune version « previous » : rollback impossible")
    client.set_registered_model_alias(name, PRODUCTION, str(prev.version))
    return str(prev.version)


def load_production_model(name: str = MODEL_NAME):
    """Charge le modèle de production (utilisable par l'API)."""
    return mlflow.sklearn.load_model(f"models:/{name}@{PRODUCTION}")


def _metrics_of(client: MlflowClient, version) -> dict:
    return client.get_run(version.run_id).data.metrics


def show(client: MlflowClient, name: str = MODEL_NAME) -> None:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        print(f"Aucune version enregistrée pour {name}")
        return
    # search_model_versions ne renvoie pas les alias : on les lit sur le
    # modèle enregistré ({alias: version}).
    by_version: dict[str, list[str]] = {}
    for alias, ver in client.get_registered_model(name).aliases.items():
        by_version.setdefault(str(ver), []).append(alias)
    for v in sorted(versions, key=lambda x: int(x.version)):
        aliases = ",".join(sorted(by_version.get(str(v.version), []))) or "-"
        m = _metrics_of(client, v)
        print(
            f"v{v.version} alias={aliases} "
            f"auc={m.get(PRIMARY_METRIC, float('nan')):.4f} "
            f"f1={m.get(SECONDARY_METRIC, float('nan')):.4f} "
            f"run={v.run_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--name", default=MODEL_NAME)
    parser.add_argument("--run-id", help="run candidat (défaut : dernier)")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cfg = load_mlflow_config(args.params)
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    client = MlflowClient()

    if args.show:
        show(client, args.name)
        return

    if args.run_id:
        run = client.get_run(args.run_id)
    else:
        run = find_final_run(client, cfg["experiment_name"])
    version = register_candidate(client, run, args.name)
    print(f"Candidat : v{version.version} (run {run.info.run_id}) -> staging")

    prod = get_by_alias(client, args.name, PRODUCTION)
    if prod is not None and str(prod.version) == str(version.version):
        print("Ce candidat est déjà en production : rien à faire.")
        return
    prod_metrics = _metrics_of(client, prod) if prod else None
    ok, reason = decide(run.data.metrics, prod_metrics, args.min_delta)
    print(f"Décision : {'PROMOTION' if ok else 'REFUS'} ({reason})")
    if ok and not args.dry_run:
        promote(client, args.name, version.version)
        print(f"v{version.version} est maintenant en production.")
    elif ok:
        print("--dry-run : aucune promotion effectuée.")
    show(client, args.name)


if __name__ == "__main__":
    main()

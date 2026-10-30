"""Prétraitement et feature engineering (tâche 3).

Le schéma décrit ici est celui observé lors de l'EDA sur le jeu
« Telecom Credit Scoring Data (Nigeria) » : 15 colonnes, cible ``default_risk``
à trois classes (low, medium, high).

Principes :
- tout ce qui s'apprend sur les données (médianes, moyennes, modalités)
  est dans
  un ``Pipeline`` scikit-learn, ajusté sur le jeu d'entraînement uniquement,
  pour éviter la fuite de données ;
- le même pipeline est réutilisé par l'API (même prétraitement en production) ;
- deux jeux de variables sont proposés (``core`` et ``with_scoring``) pour
  comparer l'effet des colonnes issues d'un scoring (voir docs/features.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# --------------------------------------------------------------------------- #
# Schéma
# --------------------------------------------------------------------------- #
TARGET = "default_risk"
TARGET_ORDER = ["low", "medium", "high"]  # codes 0, 1, 2
ID_COLUMNS = [
    "customer_id",
    "phone_number",
]  # identifiants / données personnelles
CONSTANT_COLUMNS = ["assessment_date"]  # une seule valeur dans le jeu observé
RATING_ORDER = ["poor", "fair", "good", "excellent"]

NUMERIC_CORE = [
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
]
CATEGORICAL_CORE = ["operator", "account_type"]
SCORING_NUMERIC = ["credit_score", "max_loan_amount_ngn"]
SCORING_BOOL = ["loan_eligibility"]
SCORING_ORDINAL = ["credit_rating"]

FEATURE_SETS = {
    "core": {
        "numeric": NUMERIC_CORE,
        "categorical": CATEGORICAL_CORE,
        "boolean": [],
        "ordinal": [],
    },
    "with_scoring": {
        "numeric": NUMERIC_CORE + SCORING_NUMERIC,
        "categorical": CATEGORICAL_CORE,
        "boolean": SCORING_BOOL,
        "ordinal": SCORING_ORDINAL,
    },
}
RATIO_COLUMNS = ["spend_per_recharge", "late_payments_per_year"]


def required_columns() -> list[str]:
    """Colonnes que le fichier brut doit contenir."""
    cols = set(ID_COLUMNS + CONSTANT_COLUMNS + [TARGET])
    for fs in FEATURE_SETS.values():
        for group in fs.values():
            cols.update(group)
    return sorted(cols)


# --------------------------------------------------------------------------- #
# Chargement et nettoyage
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Lit un CSV ou un Parquet et vérifie les colonnes attendues."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path)
    )
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage sans apprentissage : cible valide, doublons, ids retirés."""
    out = df.copy()
    unknown = sorted(set(out[TARGET].dropna().unique()) - set(TARGET_ORDER))
    if unknown:
        raise ValueError(
            f"Valeurs inattendues pour {TARGET} : {unknown} "
            f"(attendu : {TARGET_ORDER})"
        )
    out = out.dropna(subset=[TARGET])
    out = out.drop_duplicates()
    return out.drop(
        columns=[c for c in ID_COLUMNS + CONSTANT_COLUMNS if c in out.columns]
    )


def encode_target(y: pd.Series) -> pd.Series:
    """low -> 0, medium -> 1, high -> 2. Erreur si valeur inconnue."""
    mapping = {label: i for i, label in enumerate(TARGET_ORDER)}
    codes = y.map(mapping)
    if codes.isna().any():
        bad = sorted(y[codes.isna()].astype(str).unique())
        raise ValueError(f"Valeurs de cible inconnues : {bad}")
    return codes.astype(int)


def decode_target(codes) -> list[str]:
    return [TARGET_ORDER[int(c)] for c in codes]


# --------------------------------------------------------------------------- #
# Variables dérivées (optionnelles, à évaluer dans l'ablation MLflow)
# --------------------------------------------------------------------------- #
def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ajoute deux ratios simples, sans division par zéro possible :
    les dénominateurs sont bornés à 1 (recharge) et à 1 mois (ancienneté).
    """
    X = X.copy()
    X["spend_per_recharge"] = X["avg_monthly_spend_ngn"] / X[
        "recharge_frequency_per_month"
    ].clip(lower=1)
    X["late_payments_per_year"] = X["late_payment_count"] / (
        X["tenure_months"].clip(lower=1) / 12.0
    )
    return X


def _identity(X):
    return X


def _to_float(X):
    """Booléens -> 0.0 / 1.0 (fonction nommée : pipeline sérialisable)."""
    return X.astype(float)


# --------------------------------------------------------------------------- #
# Pipeline de prétraitement
# --------------------------------------------------------------------------- #
def build_pipeline(
    feature_set: str = "core", add_ratios: bool = False
) -> Pipeline:
    """Pipeline complet (variables dérivées éventuelles + prétraitement).

    Ajuster (``fit``) sur le jeu d'entraînement uniquement."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set inconnu : {feature_set!r}. "
            f"Choix : {list(FEATURE_SETS)}"
        )
    fs = FEATURE_SETS[feature_set]
    numeric = list(fs["numeric"]) + (RATIO_COLUMNS if add_ratios else [])

    transformers = [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            ),
            list(fs["categorical"]),
        ),
    ]
    if fs["ordinal"]:
        transformers.append(
            (
                "ord",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                categories=[RATING_ORDER],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                list(fs["ordinal"]),
            )
        )
    if fs["boolean"]:
        transformers.append(
            (
                "bool",
                FunctionTransformer(_to_float, feature_names_out="one-to-one"),
                list(fs["boolean"]),
            )
        )

    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )
    steps = [
        (
            "ratios",
            FunctionTransformer(
                add_ratio_features if add_ratios else _identity, validate=False
            ),
        ),
        ("preprocess", preprocessor),
    ]
    return Pipeline(steps).set_output(transform="pandas")


def select_features(df: pd.DataFrame, feature_set: str = "core") -> list[str]:
    fs = FEATURE_SETS[feature_set]
    return (
        list(fs["numeric"])
        + list(fs["categorical"])
        + list(fs["boolean"])
        + list(fs["ordinal"])
    )


# --------------------------------------------------------------------------- #
# Séparation des données
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame,
    feature_set: str = "core",
    test_size: float = 0.2,
    seed: int = 42,
):
    """Séparation stratifiée train/test.

    Retourne X_train, X_test, y_train, y_test (y codé 0, 1, 2)."""
    df = clean(df)
    X = df[select_features(df, feature_set)]
    y = encode_target(df[TARGET])
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sépare le jeu brut et enregistre train/test."
    )
    parser.add_argument("--input", required=True, help="CSV ou Parquet brut")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--feature-set", default="core", choices=list(FEATURE_SETS)
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_raw(args.input)
    X_train, X_test, y_train, y_test = split_data(
        df, args.feature_set, args.test_size, args.seed
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(out / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(out / "test.csv", index=False)
    print(
        f"train : {X_train.shape} | test : {X_test.shape} | "
        f"jeu de variables : {args.feature_set}"
    )
    print(
        "Répartition train :",
        y_train.value_counts(normalize=True).sort_index().round(4).to_dict(),
    )


if __name__ == "__main__":
    main()

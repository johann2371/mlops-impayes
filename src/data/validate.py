"""Validation du jeu de données brut avant l'entraînement.

Usage :
    python -m src.data.validate
Code de sortie 0 si les données sont valides, 1 sinon.
"""

import os
import sys

import pandas as pd

from src.config import ROOT, params

EXPECTED_COLUMNS = [
    "customer_id",
    "phone_number",
    "operator",
    "assessment_date",
    "credit_score",
    "credit_rating",
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
    "account_type",
    "loan_eligibility",
    "max_loan_amount_ngn",
    "default_risk",
]

# Variables du modèle (jeu "core") et cible : aucune valeur manquante tolérée.
REQUIRED_NON_NULL = [
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
    "operator",
    "account_type",
    "default_risk",
]

# Plages plausibles : minimum 0, maximum observé dans le jeu de référence.
NUMERIC_RANGES = {
    "credit_score": (0, 1000),
    "tenure_months": (0, 120),
    "avg_monthly_spend_ngn": (0, 50000),
    "recharge_frequency_per_month": (0, 30),
    "payment_consistency_score": (0, 1),
    "late_payment_count": (0, 10),
    "max_loan_amount_ngn": (0, 150000),
}

CATEGORIES = {
    "operator": {"9mobile", "MTN", "Glo", "Airtel"},
    "account_type": {"prepaid", "postpaid"},
    "default_risk": {"low", "medium", "high"},
}

MIN_ROWS = 1000


def result(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def validate_dataframe(df: pd.DataFrame) -> list:
    checks = []

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    checks.append(
        result(
            "Colonnes attendues présentes",
            not missing_cols,
            f"manquantes : {missing_cols}" if missing_cols else f"{len(EXPECTED_COLUMNS)} colonnes",
        )
    )
    checks.append(result("Nombre de lignes suffisant", len(df) >= MIN_ROWS, f"{len(df)} lignes"))

    required = [c for c in REQUIRED_NON_NULL if c in df.columns]
    nulls = df[required].isna().sum()
    nulls = nulls[nulls > 0]
    checks.append(
        result(
            "Aucune valeur manquante (variables du modèle et cible)",
            nulls.empty,
            "aucune" if nulls.empty else str(nulls.to_dict()),
        )
    )

    for col, (low, high) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            checks.append(result(f"{col} dans [{low}, {high}]", False, "colonne non numérique"))
            continue
        outside = int(((series < low) | (series > high)).sum())
        checks.append(
            result(f"{col} dans [{low}, {high}]", outside == 0, f"{outside} valeurs hors plage")
        )

    for col, allowed in CATEGORIES.items():
        if col not in df.columns:
            continue
        unexpected = sorted(set(df[col].dropna().unique()) - allowed)
        checks.append(
            result(
                f"{col} : valeurs autorisées",
                not unexpected,
                "conformes" if not unexpected else f"inattendues : {unexpected}",
            )
        )

    duplicates = int(df.duplicated().sum())
    checks.append(result("Aucune ligne dupliquée", duplicates == 0, f"{duplicates} doublons"))
    return checks


def main() -> int:
    path = ROOT / params["data"]["raw_path"]
    if not path.exists():
        print(f"Fichier introuvable : {path}")
        return 1

    checks = validate_dataframe(pd.read_csv(path))

    lines = ["| Contrôle | Résultat | Détail |", "|---|---|---|"]
    for c in checks:
        status = "OK" if c["ok"] else "ECHEC"
        lines.append(f"| {c['name']} | {status} | {c['detail']} |")
    report = "\n".join(lines)
    print(report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Validation des données\n\n" + report + "\n")

    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())

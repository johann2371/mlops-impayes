"""Simulation de dérives (tâche 10).

Le jeu ne contient aucune dérive naturelle : ``assessment_date`` est
constante et le découpage train/test est aléatoire. Pour montrer que les
alertes se déclenchent, on déforme donc des lignes **réelles** du jeu de
test. Toute sortie de ce module est une simulation et doit être
présentée comme telle (champ ``simulated`` du rapport de monitoring).

Deux dérives, deux effets attendus :
  - dérive des données : les variables changent, les étiquettes non.
    Les PSI montent ; la performance peut baisser ou non.
  - dérive de concept : les variables ne changent pas, mais le lien avec
    le risque d'impayé disparaît en partie. Les PSI restent bas ; seule
    la performance (mesurable une fois les étiquettes connues) baisse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Facteurs multiplicatifs appliqués à des variables de ``core``.
DEFAULT_SCALES = {
    "avg_monthly_spend_ngn": 1.5,
    "recharge_frequency_per_month": 0.6,
}


def simulate_data_drift(
    X: pd.DataFrame, scales: dict | None = None
) -> pd.DataFrame:
    """Copie de ``X`` dont des colonnes numériques sont multipliées."""
    scales = DEFAULT_SCALES if scales is None else scales
    missing = [c for c in scales if c not in X.columns]
    if missing:
        raise KeyError(f"Colonnes absentes de X : {missing}")
    out = X.copy()
    for col, factor in scales.items():
        out[col] = out[col] * factor
    return out


def simulate_concept_drift(
    y, fraction: float = 0.5, seed: int = 42
) -> np.ndarray:
    """Remplace ``fraction`` des étiquettes par des tirages au hasard.

    Les nouvelles étiquettes suivent la répartition observée de ``y`` et
    sont indépendantes des variables : le lien variables -> risque est
    cassé pour cette part des lignes.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction doit être comprise entre 0 et 1")
    y = np.asarray(y).copy()
    rng = np.random.RandomState(seed)
    n_changed = int(round(fraction * len(y)))
    idx = rng.choice(len(y), size=n_changed, replace=False)
    y[idx] = rng.choice(y, size=n_changed, replace=True)
    return y

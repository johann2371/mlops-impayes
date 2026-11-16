"""Dashboard de monitoring (tâche 10).

Lit ``reports/monitoring/monitoring_report.json`` produit par
``python -m src.monitoring.drift`` et l'affiche.

Lancement :
  streamlit run src/monitoring/dashboard.py

Le chemin du rapport peut être changé avec la variable d'environnement
``MONITORING_REPORT``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DEFAULT_REPORT = "reports/monitoring/monitoring_report.json"
STATUS_LABEL = {"ok": "OK", "warning": "ALERTE", "critical": "CRITIQUE"}
# Gris moyen : lisible sur le thème sombre comme sur le thème clair.
VALUE_COLOR = "#a8b0ba"


def load_report(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def overview_table(report: dict) -> pd.DataFrame:
    rows = []
    for b in report["batches"]:
        m = b.get("metrics") or {}
        rows.append(
            {
                "Lot": b["name"],
                "Nature": "simulé" if b["simulated"] else "réel",
                "Statut": STATUS_LABEL[b["status"]],
                "F1 macro": m.get("f1_macro"),
                "AUC": m.get("roc_auc_ovr_macro"),
                "Alertes": len(b["alerts"]),
                "Réentraînement": retrain_label(b["retrain"]),
            }
        )
    return pd.DataFrame(rows)


def retrain_label(retrain: dict) -> str:
    if retrain["recommended"]:
        return "recommandé"
    if retrain.get("investigate"):
        return "à investiguer"
    return "non"


def psi_table(batch: dict) -> pd.DataFrame:
    evid = batch.get("evidently") or {}
    per_col = evid.get("per_column", {})
    rows = []
    for col, value in batch["psi"].items():
        e = per_col.get(col)
        rows.append(
            {
                "Variable": col,
                "PSI": round(value, 4),
                "Evidently (distance)": (
                    None if e is None else round(e["distance"], 4)
                ),
                "Dérive Evidently": (
                    None if e is None else ("oui" if e["drifted"] else "non")
                ),
            }
        )
    return pd.DataFrame(rows)


def psi_chart(table: pd.DataFrame, thresholds: dict) -> alt.LayerChart:
    """Barres de PSI sur une échelle fixe, avec les deux seuils.

    L'échelle automatique de Streamlit étirait l'axe jusqu'au plus grand
    PSI : des valeurs de 0,0001 paraissaient aussi grandes que des
    dérives réelles. Ici l'axe va au moins jusqu'au seuil critique, la
    valeur est écrite au bout de chaque barre et les noms de variables
    ne sont pas tronqués.
    """
    warn, crit = thresholds["psi_warning"], thresholds["psi_critical"]
    top = max(crit * 1.2, float(table["PSI"].max()) * 1.1)
    color = (
        alt.when(alt.datum.PSI >= crit)
        .then(alt.value("#d62728"))
        .when(alt.datum.PSI >= warn)
        .then(alt.value("#ff7f0e"))
        .otherwise(alt.value("#4c9be8"))
    )
    y = alt.Y(
        "Variable:N",
        sort=None,
        title=None,
        axis=alt.Axis(labelLimit=400),
    )
    x = alt.X(
        "PSI:Q",
        scale=alt.Scale(domain=[0, top]),
        title="PSI",
        axis=alt.Axis(grid=False),
    )
    bars = alt.Chart(table).mark_bar().encode(
        y=y, x=x, color=color, tooltip=["Variable", "PSI"]
    )
    labels = (
        alt.Chart(table)
        .mark_text(align="left", dx=4, fontSize=11, color=VALUE_COLOR)
        .encode(y=y, x=x, text=alt.Text("PSI:Q", format=".4f"))
    )
    rules = (
        alt.Chart(
            pd.DataFrame(
                {"seuil": [warn, crit], "couleur": ["#ff7f0e", "#d62728"]}
            )
        )
        .mark_rule(strokeDash=[4, 4])
        .encode(
            x="seuil:Q", color=alt.Color("couleur:N", scale=None)
        )
    )
    return (bars + labels + rules).properties(
        height=alt.Step(28), width="container"
    )


def show_status(batch: dict) -> None:
    label = STATUS_LABEL[batch["status"]]
    text = f"Statut du lot « {batch['name']} » : {label}"
    if batch["status"] == "critical":
        st.error(text)
    elif batch["status"] == "warning":
        st.warning(text)
    else:
        st.success(text)


def show_metrics(batch: dict, baseline: dict | None) -> None:
    m = batch.get("metrics")
    if not m:
        st.info("Pas d'étiquettes pour ce lot : performance non mesurable.")
        return
    c1, c2, c3 = st.columns(3)
    for col, key, label in (
        (c1, "f1_macro", "F1 macro"),
        (c2, "roc_auc_ovr_macro", "AUC un-contre-tous"),
    ):
        value = m.get(key)
        ref = (baseline or {}).get(key)
        delta = None if value is None or ref is None else value - ref
        col.metric(
            label,
            "n/d" if value is None else f"{value:.4f}",
            None if delta is None else f"{delta:+.4f} vs production",
        )
    c3.metric("Lignes du lot", f"{batch['n_rows']:,}".replace(",", " "))


def main() -> None:
    st.set_page_config(page_title="Monitoring impayés", layout="wide")
    st.title("Monitoring du modèle de prédiction des impayés")

    path = os.environ.get("MONITORING_REPORT", DEFAULT_REPORT)
    report = load_report(path)
    if report is None:
        st.error(f"Rapport introuvable : {path}")
        st.code("python -m src.monitoring.drift")
        return

    model = report.get("model", {})
    st.caption(
        f"Modèle {model.get('name', '?')} v{model.get('version', '?')} · "
        f"rapport généré le {report['generated_at']}"
    )
    st.warning(
        "Les lots marqués « simulé » sont des lignes réelles du jeu de "
        "test volontairement déformées : le jeu ne contient aucune "
        "dérive naturelle (date d'évaluation constante)."
    )

    st.subheader("Vue d'ensemble")
    st.dataframe(overview_table(report), hide_index=True)

    names = [b["name"] for b in report["batches"]]
    choice = st.sidebar.selectbox("Lot à examiner", names)
    batch = next(b for b in report["batches"] if b["name"] == choice)

    st.header(f"Lot : {batch['name']}")
    st.write(batch["description"])
    show_status(batch)
    show_metrics(batch, report.get("baseline"))

    st.subheader("Dérive des données (PSI par variable)")
    t = report["thresholds"]
    st.caption(
        f"Seuils : alerte à {t['psi_warning']}, critique à "
        f"{t['psi_critical']}."
    )
    table = psi_table(batch)
    st.altair_chart(psi_chart(table, t), width="stretch")
    st.dataframe(table, hide_index=True)
    st.metric(
        "PSI des classes prédites", f"{batch['prediction_psi']:.4f}"
    )
    evid = batch.get("evidently") or {}
    if "error" in evid:
        st.error(f"Evidently : {evid['error']}")
    elif evid.get("html"):
        st.caption(f"Rapport Evidently détaillé : {evid['html']}")

    st.subheader("Alertes")
    if batch["alerts"]:
        st.dataframe(
            pd.DataFrame(batch["alerts"])[
                ["level", "kind", "name", "value", "threshold", "message"]
            ],
            hide_index=True,
        )
    else:
        st.success("Aucune alerte.")

    st.subheader("Réentraînement")
    retrain = batch["retrain"]
    if retrain["recommended"]:
        st.error(
            "Réentraînement recommandé : " + " ; ".join(retrain["reasons"])
        )
    elif retrain.get("investigate"):
        st.warning(
            "À investiguer (pas de baisse de performance mesurée) : "
            + " ; ".join(retrain["investigate_reasons"])
        )
    else:
        st.success("Réentraînement non nécessaire pour ce lot.")
    st.caption(retrain["note"])


if __name__ == "__main__":
    main()

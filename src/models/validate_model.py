"""Validation du modèle entraîné : seuils minimaux avant toute publication.

Usage :
    python -m src.models.validate_model
Code de sortie 0 si le modèle est valide, 1 sinon.
"""
import json
import os
import sys

from src.config import ROOT, params


def check(name: str, value: float, minimum: float) -> dict:
    return {"name": name, "value": value, "minimum": minimum, "ok": value >= minimum}


def main() -> int:
    path = ROOT / params["outputs"]["metrics_path"]
    metrics = json.loads(path.read_text(encoding="utf-8"))
    t = params["thresholds"]

    auc = metrics["test_roc_auc_ovr_macro"]
    f1 = metrics["test_f1_macro"]
    gain = f1 - metrics["test_baseline_f1_macro"]
    rows = [
        check("AUC macro (test)", auc, t["auc_macro_min"]),
        check("F1 macro (test)", f1, t["f1_macro_min"]),
        check("Gain de F1 macro sur la baseline", gain, t["min_gain_over_baseline"]),
    ]

    lines = ["| Contrôle | Valeur | Minimum | Résultat |", "|---|---|---|---|"]
    for r in rows:
        status = "OK" if r["ok"] else "ECHEC"
        lines.append(f"| {r['name']} | {r['value']:.4f} | {r['minimum']:.4f} | {status} |")
    report = "\n".join(lines)
    print(report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Validation du modèle\n\n" + report + "\n")

    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
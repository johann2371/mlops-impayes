"""Téléchargement du dataset Telecom Credit Scoring (Nigeria) depuis Hugging Face.

Usage :
    python -m src.data.ingest              # télécharge si le fichier n'existe pas
    python -m src.data.ingest --force      # retélécharge même s'il existe
    python -m src.data.ingest --sample     # crée aussi un échantillon de 5 000 lignes
"""
import argparse
from pathlib import Path

import pandas as pd

DATASET_ID = "electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria"
FILENAME = "telecom_credit_scoring_data.csv"
TARGET = "default_risk"

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / FILENAME
SAMPLE_FILE = RAW_DIR / "sample.csv"
SAMPLE_SIZE = 5000
SEED = 42


def download(force: bool = False) -> None:
    """Télécharge le CSV tel quel (mêmes octets pour tout le monde, donc même empreinte DVC)."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=DATASET_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=RAW_DIR,
        force_download=force,
    )


def check(df: pd.DataFrame) -> None:
    print(f"Lignes : {len(df)} | Colonnes : {df.shape[1]}")
    print(f"Colonnes : {df.columns.tolist()}")
    if TARGET not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET}' est absente : mauvais dataset ?")
    print(f"Répartition de '{TARGET}' :")
    print(df[TARGET].value_counts(normalize=True).round(4))


def main(force: bool = False, sample: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and not force:
        print(f"{FILENAME} existe déjà (utilise --force pour retélécharger).")
    else:
        print(f"Téléchargement de {DATASET_ID} ...")
        download(force=force)
        print(f"Enregistré : {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    check(df)

    if sample:
        df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).to_csv(
            SAMPLE_FILE, index=False
        )
        print(f"Échantillon enregistré : {SAMPLE_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    main(force=args.force, sample=args.sample)
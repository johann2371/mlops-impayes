from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)
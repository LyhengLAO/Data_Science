from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else ROOT /  "configs" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)

def abspath(relative_path: str | Path) -> Path:
    return ROOT / relative_path
import yaml
from pathlib import Path


def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

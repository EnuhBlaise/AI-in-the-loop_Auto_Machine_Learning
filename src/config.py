"""Configuration loading and merging utilities."""

import copy
import os
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using dot-separated key."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config. Defaults to config/base.yaml.
        overrides: Dict of dot-separated key overrides (e.g. {"training.lr": 0.01}).

    Returns:
        Merged configuration dict.
    """
    project_root = Path(__file__).resolve().parent.parent

    if config_path is None:
        config_path = project_root / "config" / "base.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    # Resolve relative data path
    data_path = Path(config["data"]["path"])
    if not data_path.is_absolute():
        config["data"]["path"] = str(project_root / data_path)

    # Apply env var overrides
    if os.environ.get("BIOML_LLM_MODEL"):
        config["llm"]["model"] = os.environ["BIOML_LLM_MODEL"]
    if os.environ.get("HF_TOKEN"):
        config["llm"]["hf_token"] = os.environ["HF_TOKEN"]
    if os.environ.get("BIOML_TIME_BUDGET"):
        config["experiment"]["time_budget"] = int(os.environ["BIOML_TIME_BUDGET"])
    if os.environ.get("BIOML_MAX_EXPERIMENTS"):
        config["autonomous"]["max_experiments"] = int(os.environ["BIOML_MAX_EXPERIMENTS"])

    return config


def load_search_space(path: str | Path | None = None) -> dict:
    """Load the hyperparameter search space definition."""
    project_root = Path(__file__).resolve().parent.parent
    if path is None:
        path = project_root / "config" / "search_space.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["search_space"]

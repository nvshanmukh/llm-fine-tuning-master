"""
Configuration loading utilities.
Supports YAML configs via OmegaConf, with optional .env loading.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """
    Load a YAML config using OmegaConf, with lightweight ``defaults`` support.

    If the file contains a top-level ``defaults: [- base, - other]`` list, each
    named sibling YAML (``configs/<name>.yaml``) is loaded first and merged in
    order, then the current file's own keys override them. This mirrors the
    Hydra-style composition the configs were written for without pulling in
    Hydra's full runtime.

    Raises:
        FileNotFoundError: If the config file (or a referenced default) is missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = cast(DictConfig, OmegaConf.load(config_path))
    defaults = raw.pop("defaults", None) if isinstance(raw, DictConfig) else None

    merged: DictConfig = OmegaConf.create({})  # type: ignore[assignment]
    for entry in defaults or []:
        name = str(entry).lstrip("- ").strip()
        if not name or name == "_self_":
            continue
        dep_path = config_path.parent / f"{name}.yaml"
        if not dep_path.exists():
            raise FileNotFoundError(f"Config '{config_path}' references missing default: {dep_path}")
        merged = cast(DictConfig, OmegaConf.merge(merged, load_config(dep_path)))

    merged = cast(DictConfig, OmegaConf.merge(merged, raw))
    logger.info(f"Loaded config from: {config_path}" + (f" (+defaults {defaults})" if defaults else ""))
    return merged


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple OmegaConf configs, later configs override earlier ones.

    Args:
        *configs: Variable number of DictConfig objects.

    Returns:
        Merged DictConfig.
    """
    return cast(DictConfig, OmegaConf.merge(*configs))


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf DictConfig to a plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore


def resolve_mlflow_uri(configured: str | None = None) -> str:
    """
    Resolve the MLflow tracking URI.

    Precedence: MLFLOW_TRACKING_URI env var > ``configured`` value > sqlite default.
    Bare local paths (``./mlruns``) are rewritten to a SQLite DB because recent
    MLflow refuses the legacy filesystem store by default.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI") or configured or "sqlite:///mlflow.db"
    if uri in ("./mlruns", "mlruns", "file:./mlruns", "./mlruns/"):
        uri = "sqlite:///mlflow.db"
    return uri


def load_env_overrides() -> dict[str, str]:
    """
    Load environment variable overrides.
    Reads from a .env file if present.

    Returns:
        Dictionary of environment variable values.
    """
    env_vars = {}
    env_file = Path(".env")
    if env_file.exists():
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()
                    os.environ.setdefault(key.strip(), value.strip())
        logger.info(f"Loaded {len(env_vars)} env vars from .env")
    return env_vars

"""
Configuration loading utilities.
Supports YAML configs via OmegaConf, with optional .env loading.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """
    Load a YAML configuration file using OmegaConf.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        OmegaConf DictConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from: {config_path}")
    return cfg


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple OmegaConf configs, later configs override earlier ones.

    Args:
        *configs: Variable number of DictConfig objects.

    Returns:
        Merged DictConfig.
    """
    merged = OmegaConf.merge(*configs)
    return merged


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf DictConfig to a plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore


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

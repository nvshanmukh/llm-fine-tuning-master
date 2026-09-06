"""
Reproducibility utilities — set random seeds across all relevant libraries.
"""
from __future__ import annotations

import os
import random

import numpy as np
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (if available).
    Also sets PYTHONHASHSEED for full reproducibility.

    Args:
        seed: Integer seed value. Default: 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensures deterministic ops (may reduce performance slightly)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Seeds set: Python={seed}, NumPy={seed}, PyTorch={seed}")
    except ImportError:
        logger.warning("PyTorch not installed — only Python/NumPy seeds set")

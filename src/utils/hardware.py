"""
Hardware introspection utilities.
Detects GPU/CPU, VRAM, RAM, and returns a hardware summary dict.
"""
from __future__ import annotations

import platform

from loguru import logger


def get_hardware_info() -> dict[str, str]:
    """
    Collect system hardware information for experiment logging.

    Returns:
        Dict with keys: gpu_name, gpu_vram_gb, cpu, ram_gb, cuda_version, torch_version.
    """
    info: dict[str, str] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
    }

    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        info["ram_gb"] = f"{total_ram:.1f}"
    except ImportError:
        info["ram_gb"] = "unknown"

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["gpu_vram_gb"] = f"{vram:.1f}"
            info["gpu_count"] = str(torch.cuda.device_count())
        else:
            info["cuda_version"] = "N/A"
            info["gpu_name"] = "CPU only"
            info["gpu_vram_gb"] = "0"
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = "false"

    return info


def get_gpu_memory_usage() -> dict[str, float]:
    """
    Get current GPU memory usage in GB.

    Returns:
        Dict with allocated_gb and reserved_gb. Returns zeros on CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            return {"allocated_gb": round(allocated, 3), "reserved_gb": round(reserved, 3)}
    except ImportError:
        pass
    return {"allocated_gb": 0.0, "reserved_gb": 0.0}


def log_hardware_info() -> dict[str, str]:
    """Log and return hardware info."""
    info = get_hardware_info()
    logger.info("=" * 60)
    logger.info("Hardware Summary:")
    for k, v in info.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    return info

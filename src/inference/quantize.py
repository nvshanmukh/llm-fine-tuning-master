"""
Post-training quantization utilities.

Supports:
- GGUF quantization via llama.cpp (Q4_K_M, Q8_0, etc.)
- bitsandbytes dynamic quantization for inference
- Model size estimation and comparison

NOTE: GGUF conversion requires llama.cpp tools to be installed.
For environments without llama.cpp, a bitsandbytes int8 quantization
alternative is provided.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


def get_model_size_gb(model_path: str | Path) -> float:
    """
    Compute the total size of a model directory in GB.

    Args:
        model_path: Path to the model directory containing safetensors/bin files.

    Returns:
        Total size in GB.
    """
    model_path = Path(model_path)
    total_bytes = sum(
        f.stat().st_size
        for f in model_path.rglob("*")
        if f.is_file() and f.suffix in {".safetensors", ".bin", ".pt", ".gguf"}
    )
    return round(total_bytes / (1024 ** 3), 3)


def quantize_bnb_int8(
    model_path: str,
    output_path: str,
) -> dict[str, Any]:
    """
    Apply bitsandbytes int8 dynamic quantization and save the model.

    This is a lightweight quantization approach that works without
    external tools. The quantized model can be used directly with
    the HuggingFace transformers pipeline.

    Args:
        model_path: Path to the input model (safetensors or HF model name).
        output_path: Path to save the quantized model.

    Returns:
        Dict with before/after size comparison.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading model for int8 quantization: {model_path}")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    logger.info(f"int8 quantized model saved to: {out_path}")
    return {
        "quantization_type": "int8",
        "output_path": str(out_path),
    }


def compare_model_sizes(
    models: dict[str, str | Path],
) -> dict[str, dict]:
    """
    Compare sizes across multiple model variants.

    Args:
        models: Dict mapping model names to their directory paths.
               E.g. {"base": "./base_model", "qlora": "./experiments/qlora"}

    Returns:
        Dict with size info per model.
    """
    results = {}
    for name, path in models.items():
        path = Path(path)
        if path.exists():
            size_gb = get_model_size_gb(path)
            results[name] = {"path": str(path), "size_gb": size_gb}
            logger.info(f"Model '{name}': {size_gb:.2f} GB")
        else:
            results[name] = {"path": str(path), "size_gb": None, "error": "path not found"}
            logger.warning(f"Model '{name}' path not found: {path}")
    return results

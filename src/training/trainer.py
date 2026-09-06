"""
Core training module for LoRA and QLoRA fine-tuning.

Supports:
- LoRA fine-tuning (full precision, bfloat16)
- QLoRA fine-tuning (4-bit NF4 quantization via bitsandbytes)
- MLflow experiment tracking
- Gradient checkpointing
- HuggingFace SFTTrainer from TRL

All hyperparameters are driven by OmegaConf config (never hardcoded).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig


def build_bnb_config(cfg: DictConfig):
    """
    Build a BitsAndBytesConfig for QLoRA training.

    Args:
        cfg: Model configuration section (cfg.model).

    Returns:
        BitsAndBytesConfig or None if quantization is disabled.
    """
    import torch
    from transformers import BitsAndBytesConfig

    if not cfg.get("load_in_4bit", False):
        return None

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )
    logger.info(
        f"QLoRA: 4-bit NF4 quantization enabled, "
        f"compute_dtype={cfg.get('bnb_4bit_compute_dtype', 'bfloat16')}, "
        f"double_quant={cfg.get('bnb_4bit_use_double_quant', True)}"
    )
    return bnb_config


def load_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple:
    """
    Load the base model and tokenizer.

    Handles:
    - Standard loading (LoRA)
    - 4-bit quantized loading (QLoRA)

    Args:
        cfg: Full config (uses cfg.model).

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg.model
    model_name = model_cfg.model_name_or_path

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    bnb_config = build_bnb_config(model_cfg)

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )

    if bnb_config is not None:
        # Required for QLoRA: prepare model for k-bit training
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
        )
        logger.info("Model prepared for k-bit training (QLoRA)")

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        token=os.environ.get("HF_TOKEN"),
    )
    # Qwen2 uses a specific pad token strategy
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Critical for causal LM training

    # Log model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params / 1e6:.1f}M")

    return model, tokenizer


def build_peft_config(cfg: DictConfig):
    """
    Build a LoraConfig from the peft section of the config.

    Args:
        cfg: Full config (uses cfg.peft).

    Returns:
        PEFT LoraConfig.
    """
    from peft import LoraConfig, TaskType

    peft_cfg = cfg.peft
    target_modules = list(peft_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    lora_config = LoraConfig(
        r=peft_cfg.get("r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias=peft_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    logger.info(
        f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}, modules={target_modules}"
    )
    return lora_config


def count_trainable_params(model) -> dict[str, int]:
    """
    Count trainable vs total parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dict with trainable_params, total_params, trainable_pct.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 4),
    }


def build_training_args(cfg: DictConfig, experiment_run_dir: Path):
    """
    Build HuggingFace TrainingArguments from config.

    Args:
        cfg: Full config (uses cfg.training).
        experiment_run_dir: Directory to save checkpoints.

    Returns:
        TrainingArguments instance.
    """
    from transformers import TrainingArguments

    train_cfg = cfg.training
    args = TrainingArguments(
        output_dir=str(experiment_run_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 50),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # MLflow handled via callback
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=cfg.project.get("seed", 42),
    )
    return args


def run_training(
    cfg: DictConfig,
    train_dataset,
    eval_dataset,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Main training entrypoint. Runs LoRA or QLoRA training based on cfg.

    Args:
        cfg: Full OmegaConf config.
        train_dataset: HuggingFace Dataset for training.
        eval_dataset: HuggingFace Dataset for validation.
        experiment_name: Override for MLflow experiment name.

    Returns:
        Dict containing training results and metrics.
    """
    import mlflow
    from peft import get_peft_model
    from trl import SFTTrainer

    from src.data.prompt_template import format_for_training
    from src.utils.hardware import get_gpu_memory_usage, log_hardware_info

    # Set up experiment directory
    exp_name = experiment_name or cfg.training.get("experiment_name", "run")
    run_dir = Path(cfg.training.get("output_dir", "./experiments")) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log hardware info
    hw_info = log_hardware_info()

    # Set up MLflow
    mlflow_uri = cfg.project.get("mlflow_tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow_exp = cfg.project.get("mlflow_experiment_name", "finance-llm")
    mlflow.set_experiment(mlflow_exp)

    with mlflow.start_run(run_name=exp_name) as mlflow_run:
        # Log config
        from src.utils.config_utils import config_to_dict
        config_dict = config_to_dict(cfg)
        mlflow.log_params({str(k): v for k, v in _flatten_dict(config_dict).items() if len(str(v)) < 250})
        mlflow.log_dict(config_dict, "config.json")

        # Log hardware
        mlflow.log_params({f"hw_{k}": v for k, v in hw_info.items()})

        # 1. Load model + tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg)

        # 2. Build PEFT config
        lora_config = build_peft_config(cfg)

        # 3. Wrap model with LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        param_counts = count_trainable_params(model)
        mlflow.log_params(param_counts)
        logger.info(
            f"Trainable params: {param_counts['trainable_params'] / 1e6:.2f}M / "
            f"{param_counts['total_params'] / 1e6:.0f}M "
            f"({param_counts['trainable_pct']:.2f}%)"
        )

        # 4. Build TrainingArguments
        training_args = build_training_args(cfg, run_dir)

        # 5. Format datasets
        def _format_dataset(example: dict) -> dict:
            return {"text": format_for_training(example)}

        train_ds = train_dataset.map(_format_dataset)
        eval_ds = eval_dataset.map(_format_dataset)

        # 6. Log memory before training
        gpu_mem_before = get_gpu_memory_usage()
        mlflow.log_metrics({f"gpu_mem_before_{k}": v for k, v in gpu_mem_before.items()})

        # 7. Train
        logger.info("Starting training...")
        train_start = time.time()

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=None,  # already applied above
            dataset_text_field="text",
            max_seq_length=cfg.training.get("max_seq_length", 512),
            dataset_num_proc=1,
        )

        train_result = trainer.train()
        train_elapsed = time.time() - train_start

        # 8. Log post-training metrics
        gpu_mem_after = get_gpu_memory_usage()
        mlflow.log_metrics({f"gpu_mem_after_{k}": v for k, v in gpu_mem_after.items()})
        mlflow.log_metric("training_time_seconds", train_elapsed)
        mlflow.log_metrics(train_result.metrics)

        # 9. Save model
        final_model_path = run_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        logger.info(f"Model saved to: {final_model_path}")
        mlflow.log_param("final_model_path", str(final_model_path))

        logger.info(f"Training complete in {train_elapsed / 60:.1f} minutes")

        return {
            "run_id": mlflow_run.info.run_id,
            "experiment_name": exp_name,
            "model_path": str(final_model_path),
            "train_metrics": train_result.metrics,
            "training_time_s": train_elapsed,
            "param_counts": param_counts,
            "gpu_memory": gpu_mem_after,
        }


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary for MLflow param logging."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key))
        else:
            result[key] = v
    return result

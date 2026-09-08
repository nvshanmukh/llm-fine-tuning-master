"""
Core training module for LoRA fine-tuning.

Supports:
- LoRA fine-tuning (full precision: bf16 / fp16 / fp32)
- Response-only loss masking (completion-only SFT)
- MLflow experiment tracking (params, loss curves, memory, artifacts)
- Gradient checkpointing

All hyperparameters are driven by OmegaConf config (never hardcoded).

DESIGN
------
Training uses the plain HuggingFace ``Trainer`` plus PEFT. Tokenization and
response-only label masking are done explicitly in ``src.training.data`` so the
code does not depend on TRL's repeatedly-changing SFTTrainer API.

4-bit (QLoRA-style) *training* is not a shipped config -- see docs/VALIDATION.md.
The ``load_in_4bit`` support below is retained for anyone who installs
bitsandbytes on a CUDA host and sets the flag manually: it aborts with an
explicit error rather than silently falling back to CPU / full precision.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig


def _assert_qlora_supported() -> None:
    """Fail loudly if 4-bit QLoRA training cannot actually run here."""
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("QLoRA requires PyTorch to be installed.") from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA (load_in_4bit) requires a CUDA GPU. No GPU is visible to torch. "
            "Use configs/lora.yaml on CPU/GPU full precision, or run QLoRA on a CUDA host."
        )
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "QLoRA requires the 'bitsandbytes' package with a CUDA build. "
            "Install it on a CUDA host (see requirements.txt)."
        ) from e


def build_bnb_config(cfg: DictConfig):
    """Build a BitsAndBytesConfig for QLoRA training, or None when disabled."""
    import torch
    from transformers import BitsAndBytesConfig

    if not cfg.get("load_in_4bit", False):
        return None

    _assert_qlora_supported()

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
        f"QLoRA: 4-bit {cfg.get('bnb_4bit_quant_type', 'nf4')} enabled, "
        f"compute_dtype={cfg.get('bnb_4bit_compute_dtype', 'bfloat16')}, "
        f"double_quant={cfg.get('bnb_4bit_use_double_quant', True)}"
    )
    return bnb_config


def _resolve_precision(cfg: DictConfig) -> tuple[bool, bool]:
    """Return (bf16, fp16), downgrading bf16->fp16 when the GPU lacks bf16 support."""
    import torch

    want_bf16 = bool(cfg.training.get("bf16", True))
    want_fp16 = bool(cfg.training.get("fp16", False))
    if want_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning("GPU does not support bf16; falling back to fp16.")
        return False, True
    if want_bf16 and not torch.cuda.is_available():
        logger.warning("No CUDA GPU; disabling bf16/fp16 mixed precision.")
        return False, False
    return want_bf16, want_fp16


def load_model_and_tokenizer(cfg: DictConfig) -> tuple:
    """Load the base model and tokenizer (standard or 4-bit quantized)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg.model
    model_name = model_cfg.model_name_or_path
    revision = model_cfg.get("revision")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)
    bnb_config = build_bnb_config(model_cfg)

    logger.info(f"Loading model: {model_name} @ {revision or 'main'}")
    load_kwargs = dict(
        revision=revision,
        quantization_config=bnb_config,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        device_map="auto" if torch.cuda.is_available() else None,
        token=os.environ.get("HF_TOKEN"),
    )
    dtype = torch_dtype if bnb_config is None else None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **load_kwargs)
    except TypeError:  # transformers < 4.55
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **load_kwargs)

    if bnb_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.training.get("gradient_checkpointing", True)
        )
        logger.info("Model prepared for k-bit training (QLoRA)")

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params / 1e6:.1f}M")
    return model, tokenizer


def build_peft_config(cfg: DictConfig):
    """Build a LoraConfig from cfg.peft."""
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
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 4) if total else 0.0,
    }


def _warmup_steps(cfg: DictConfig, num_train_examples: int | None) -> int:
    """Resolve warmup steps (transformers 5.x dropped ``warmup_ratio``)."""
    train_cfg = cfg.training
    if train_cfg.get("warmup_steps") is not None:
        return int(train_cfg.warmup_steps)
    ratio = float(train_cfg.get("warmup_ratio", 0.03))
    if not num_train_examples or ratio <= 0:
        return 0
    eff_batch = (
        int(train_cfg.get("per_device_train_batch_size", 4))
        * int(train_cfg.get("gradient_accumulation_steps", 4))
    )
    max_steps = int(train_cfg.get("max_steps", -1))
    if max_steps > 0:
        total = max_steps
    else:
        steps_per_epoch = max(1, num_train_examples // max(1, eff_batch))
        total = steps_per_epoch * int(train_cfg.get("num_train_epochs", 3))
    return max(0, int(total * ratio))


def build_training_args(cfg: DictConfig, experiment_run_dir: Path,
                        num_train_examples: int | None = None):
    """Build HuggingFace TrainingArguments from cfg.training."""
    from transformers import TrainingArguments

    train_cfg = cfg.training
    bf16, fp16 = _resolve_precision(cfg)
    max_steps = int(train_cfg.get("max_steps", -1))

    return TrainingArguments(
        output_dir=str(experiment_run_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=_warmup_steps(cfg, num_train_examples),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # MLflow handled explicitly via callback + run_training
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=cfg.project.get("seed", 42),
    )


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key))
        else:
            result[key] = v
    return result


def run_training(
    cfg: DictConfig,
    train_dataset,
    eval_dataset,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Main training entrypoint. Runs LoRA or QLoRA training based on cfg."""
    import mlflow
    from peft import get_peft_model
    from transformers import Trainer

    from src.training.callbacks import MLflowDetailedCallback
    from src.training.data import CausalCollator, build_tokenized_dataset
    from src.utils.config_utils import config_to_dict
    from src.utils.hardware import get_gpu_memory_usage, log_hardware_info

    exp_name = experiment_name or cfg.training.get("experiment_name", "run")
    run_dir = Path(cfg.training.get("output_dir", "./experiments")) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    hw_info = log_hardware_info()

    from src.utils.config_utils import resolve_mlflow_uri

    mlflow_uri = resolve_mlflow_uri(cfg.project.get("mlflow_tracking_uri"))
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg.project.get("mlflow_experiment_name", "finance-llm"))
    logger.info(f"MLflow tracking URI: {mlflow_uri}")

    with mlflow.start_run(run_name=exp_name) as mlflow_run:
        config_dict = config_to_dict(cfg)
        mlflow.log_params(
            {str(k): v for k, v in _flatten_dict(config_dict).items() if len(str(v)) < 250}
        )
        mlflow.log_dict(config_dict, "config.json")
        mlflow.log_params({f"hw_{k}": v for k, v in hw_info.items()})

        model, tokenizer = load_model_and_tokenizer(cfg)
        lora_config = build_peft_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        param_counts = count_trainable_params(model)
        mlflow.log_params(param_counts)
        logger.info(
            f"Trainable params: {param_counts['trainable_params'] / 1e6:.2f}M / "
            f"{param_counts['total_params'] / 1e6:.0f}M ({param_counts['trainable_pct']:.2f}%)"
        )

        training_args = build_training_args(cfg, run_dir, num_train_examples=len(train_dataset))
        max_seq_length = int(cfg.training.get("max_seq_length", 512))

        # Response-only loss: prompt tokens are masked to -100 so the model is
        # scored only on the answer, not on memorising the prompt.
        train_ds = build_tokenized_dataset(train_dataset, tokenizer, max_seq_length)
        eval_ds = build_tokenized_dataset(eval_dataset, tokenizer, max_seq_length)
        logger.info(f"Tokenized: train={len(train_ds):,}, eval={len(eval_ds):,}")
        collator = CausalCollator(tokenizer=tokenizer)

        mlflow.log_metrics({f"gpu_mem_before_{k}": v for k, v in get_gpu_memory_usage().items()})

        logger.info("Starting training...")
        train_start = time.time()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[MLflowDetailedCallback()],
        )
        train_result = trainer.train()
        train_elapsed = time.time() - train_start

        gpu_mem_after = get_gpu_memory_usage()
        mlflow.log_metrics({f"gpu_mem_after_{k}": v for k, v in gpu_mem_after.items()})
        mlflow.log_metric("training_time_seconds", train_elapsed)
        mlflow.log_metrics(
            {k: float(v) for k, v in train_result.metrics.items() if isinstance(v, (int, float))}
        )

        try:
            eval_metrics = trainer.evaluate()
            mlflow.log_metrics(
                {f"final_{k}": float(v) for k, v in eval_metrics.items()
                 if isinstance(v, (int, float))}
            )
        except Exception as e:
            logger.warning(f"Final evaluate() failed: {e}")
            eval_metrics = {}

        final_model_path = run_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        adapter_size_mb = round(
            sum(f.stat().st_size for f in final_model_path.rglob("*") if f.is_file())
            / (1024 ** 2), 2,
        )
        mlflow.log_param("final_model_path", str(final_model_path))
        mlflow.log_metric("adapter_size_mb", adapter_size_mb)
        mlflow.log_artifacts(str(final_model_path), artifact_path="final_model")
        logger.info(f"Model saved to: {final_model_path} ({adapter_size_mb} MB)")
        logger.info(f"Training complete in {train_elapsed / 60:.1f} minutes")

        return {
            "run_id": mlflow_run.info.run_id,
            "experiment_name": exp_name,
            "model_path": str(final_model_path),
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "training_time_s": train_elapsed,
            "param_counts": param_counts,
            "adapter_size_mb": adapter_size_mb,
            "gpu_memory": gpu_mem_after,
            "status": "completed",
        }

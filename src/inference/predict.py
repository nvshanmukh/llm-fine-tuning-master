"""
Inference module for the Finance LLM fine-tuning project.

Supports:
- Base model inference (no adapter)
- LoRA/QLoRA adapter inference (via PEFT merge or adapter loading)
- Configurable generation parameters
- Latency measurement
- Batch inference
- Structured result objects

Designed to be used by:
1. The CLI scripts (scripts/inference.py)
2. The FastAPI service (src/api/main.py)
3. The evaluation pipeline (scripts/evaluate.py)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class GenerationConfig:
    """Generation hyperparameters for inference."""
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    # If do_sample=False, greedy decoding is used (ignores temp/top_p)


@dataclass
class InferenceResult:
    """Structured output from a single inference call."""
    prompt: str
    response: str
    model_id: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


class FinanceLLMPredictor:
    """
    Inference engine for the Finance LLM.

    Handles model/tokenizer loading, optional adapter loading,
    and generation with latency measurement.

    Usage:
        predictor = FinanceLLMPredictor(
            model_path="Qwen/Qwen2.5-1.5B",
            adapter_path="./experiments/qlora/final_model",  # optional
        )
        result = predictor.generate("What is a mutual fund?")
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype_str: str = "auto",
        device_map: str = "auto",
        model_id: str = "unknown",
    ) -> None:
        """
        Initialize the predictor and load the model.

        Args:
            model_path: HuggingFace model name or local path to base model.
            adapter_path: Optional path to LoRA/QLoRA adapter weights.
            load_in_4bit: Enable 4-bit quantization (QLoRA inference).
            load_in_8bit: Enable 8-bit quantization.
            torch_dtype_str: PyTorch dtype ('auto', 'bfloat16', 'float16').
            device_map: Device mapping strategy.
            model_id: Human-readable model identifier for logging.
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model_id = model_id
        self._model: Any = None
        self._tokenizer: Any = None

        self._load_model(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype_str=torch_dtype_str,
            device_map=device_map,
        )

    def _load_model(self, load_in_4bit, load_in_8bit, torch_dtype_str, device_map) -> None:
        """Load model and tokenizer, optionally applying adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(torch_dtype_str, "auto")

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading model: {self.model_path}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype if bnb_config is None else None,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Load adapter if provided
        if self.adapter_path:
            from peft import PeftModel
            logger.info(f"Loading adapter: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            self._model = self._model.merge_and_unload()  # fuse for faster inference
            logger.info("Adapter merged into base model")

        self._model.eval()

        logger.info(f"Loading tokenizer: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Model ready: {self.model_id}")

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> InferenceResult:
        """
        Generate a response for a single prompt.

        Args:
            prompt: Formatted prompt string (use format_for_inference from prompt_template).
            config: Generation configuration. Uses defaults if None.

        Returns:
            InferenceResult with response, latency, and token counts.
        """
        import torch

        if config is None:
            config = GenerationConfig()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        input_token_count = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Decode only the newly generated tokens (not the prompt)
        new_tokens = output_ids[0][input_token_count:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        output_token_count = len(new_tokens)

        return InferenceResult(
            prompt=prompt,
            response=response,
            model_id=self.model_id,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            generation_config=config,
        )

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        show_progress: bool = True,
    ) -> list[InferenceResult]:
        """
        Generate responses for a batch of prompts (one at a time).

        Args:
            prompts: List of formatted prompt strings.
            config: Generation configuration.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of InferenceResult objects.
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(prompts, desc=f"Generating [{self.model_id}]") if show_progress else prompts
        for prompt in iterator:
            result = self.generate(prompt, config)
            results.append(result)
        return results

    @property
    def model_size_mb(self) -> float:
        """Estimate model size in MB from parameter byte count."""
        total_bytes = sum(
            p.numel() * p.element_size()
            for p in self._model.parameters()
        )
        return round(total_bytes / (1024 ** 2), 1)

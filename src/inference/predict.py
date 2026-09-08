"""
Inference module for the Finance LLM fine-tuning project.

Supports:
- Base model inference (no adapter)
- LoRA/QLoRA adapter inference (adapter kept attached; merged only for
  full-precision bases where merging is numerically safe)
- Configurable generation parameters
- Latency measurement (with CUDA synchronisation and optional warm-up)
- Batch inference
- Structured result objects

Designed to be used by:
1. The CLI scripts (scripts/inference.py)
2. The FastAPI service (src/api/main.py)
3. The evaluation pipeline (scripts/evaluate.py)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    max_input_tokens: int = 1024
    # If do_sample=False, greedy decoding is used (temperature/top_p ignored).


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
        device_map: str | None = "auto",
        model_id: str = "unknown",
        merge_adapter: bool | None = None,
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
            merge_adapter: Whether to fuse the adapter into the base weights.
                Defaults to True for full-precision bases and False for
                quantized bases (merging into 4-/8-bit weights is not
                supported by PEFT and would corrupt the model).
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.adapter_merged = False
        self._model: Any = None
        self._tokenizer: Any = None

        if merge_adapter is None:
            merge_adapter = not (load_in_4bit or load_in_8bit)
        self._merge_adapter = merge_adapter

        self._validate_adapter_compatibility()
        self._load_model(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype_str=torch_dtype_str,
            device_map=device_map,
        )

    def _validate_adapter_compatibility(self) -> None:
        """Check that the adapter (if any) was trained for this base model."""
        if not self.adapter_path:
            return
        cfg_path = Path(self.adapter_path) / "adapter_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"No adapter_config.json in {self.adapter_path!r} -- not a PEFT adapter directory"
            )
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt adapter_config.json: {e}") from e
        trained_base = cfg.get("base_model_name_or_path")
        if trained_base and trained_base != self.model_path:
            logger.warning(
                f"Adapter was trained on base {trained_base!r} but inference base is "
                f"{self.model_path!r}. Proceeding, but results may be wrong."
            )

    def _load_model(self, load_in_4bit, load_in_8bit, torch_dtype_str, device_map) -> None:
        """Load model and tokenizer, optionally applying adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(torch_dtype_str, "auto")

        bnb_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading model: {self.model_path}")
        load_kwargs = dict(
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        dtype = torch_dtype if bnb_config is None else None
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=dtype, **load_kwargs
            )
        except TypeError:  # transformers < 4.55 uses torch_dtype
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=dtype, **load_kwargs
            )

        # Load adapter if provided
        if self.adapter_path:
            from peft import PeftModel

            logger.info(f"Loading adapter: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            if self._merge_adapter:
                self._model = self._model.merge_and_unload()
                self.adapter_merged = True
                logger.info("Adapter merged into base model")
            else:
                logger.info("Adapter attached (not merged -- quantized base)")

        self._model.eval()

        logger.info(f"Loading tokenizer: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Model ready: {self.model_id}")

    def _sync(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def warmup(self, config: GenerationConfig | None = None) -> None:
        """Run one short throwaway generation so latency numbers exclude lazy init."""
        base = config or GenerationConfig()
        warm = GenerationConfig(
            max_new_tokens=8, do_sample=base.do_sample,
            temperature=base.temperature, top_p=base.top_p,
            repetition_penalty=base.repetition_penalty,
        )
        try:
            self.generate("<|im_start|>user\nwarmup<|im_end|>\n<|im_start|>assistant\n", warm)
        except Exception as e:
            logger.warning(f"Warm-up generation failed (non-fatal): {e}")

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> InferenceResult:
        """
        Generate a response for a single prompt.

        Latency covers tokenization + generation + decoding for a single
        example (batch size 1), measured with CUDA synchronisation.
        """
        import torch

        if config is None:
            config = GenerationConfig()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_input_tokens,
        ).to(self._model.device)

        input_token_count = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": config.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if config.do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p

        self._sync()
        start_time = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        self._sync()
        latency_ms = (time.perf_counter() - start_time) * 1000

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
        """Generate responses for a list of prompts (sequentially, batch size 1)."""
        try:
            from tqdm import tqdm
        except ImportError:  # tqdm is optional
            def tqdm(x, **_):  # type: ignore
                return x

        results = []
        iterator = tqdm(prompts, desc=f"Generating [{self.model_id}]") if show_progress else prompts
        for prompt in iterator:
            results.append(self.generate(prompt, config))
        return results

    @property
    def model_size_mb(self) -> float:
        """
        In-memory size of the model parameters in MB.

        Note: for 4-/8-bit quantized models this reflects the packed
        parameter bytes actually held in memory, not the notional
        full-precision size.
        """
        total_bytes = sum(
            p.numel() * p.element_size()
            for p in self._model.parameters()
        )
        return round(total_bytes / (1024 ** 2), 1)

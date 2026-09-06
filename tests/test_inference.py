"""
Smoke tests for the inference pipeline.

Validates prompt formatting, GenerationConfig, InferenceResult,
and FinanceLLMPredictor interface — all without loading a real model.
These tests run on any machine, including CPU-only environments.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.data.prompt_template import (
    SYSTEM_PROMPT,
    build_formatted_example,
    format_for_inference,
    format_for_training,
)
from src.inference.predict import GenerationConfig, InferenceResult

# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------

class TestGenerationConfig:
    def test_default_values(self):
        config = GenerationConfig()
        assert config.max_new_tokens == 256
        assert config.temperature == 0.1
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.1
        assert config.do_sample is True

    def test_custom_values_assigned_correctly(self):
        config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            do_sample=False,
        )
        assert config.max_new_tokens == 128
        assert config.temperature == 0.7
        assert config.top_p == 0.8
        assert config.do_sample is False

    def test_greedy_decoding_config(self):
        """do_sample=False should be valid (greedy decoding)."""
        config = GenerationConfig(do_sample=False)
        assert config.do_sample is False


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------

class TestInferenceResult:
    def test_creation_with_required_fields(self):
        result = InferenceResult(
            prompt="test prompt text",
            response="The answer to the question.",
            model_id="test-model",
            latency_ms=123.4,
            input_tokens=25,
            output_tokens=8,
        )
        assert result.prompt == "test prompt text"
        assert result.response == "The answer to the question."
        assert result.model_id == "test-model"
        assert result.latency_ms == pytest.approx(123.4)
        assert result.input_tokens == 25
        assert result.output_tokens == 8

    def test_default_generation_config(self):
        result = InferenceResult(
            prompt="p", response="r", model_id="m",
            latency_ms=10.0, input_tokens=5, output_tokens=3,
        )
        assert isinstance(result.generation_config, GenerationConfig)


# ---------------------------------------------------------------------------
# Prompt template (inference-focused)
# ---------------------------------------------------------------------------

class TestInferencePromptFormatting:
    """Verify that inference prompts are correctly structured for Qwen2.5."""

    def test_qwen_chatml_tokens_present(self):
        ex = {"instruction": "What is a bond?", "input": ""}
        prompt = format_for_inference(ex)
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "<|im_end|>" in prompt

    def test_system_prompt_is_included(self):
        ex = {"instruction": "Define GDP.", "input": ""}
        prompt = format_for_inference(ex)
        assert SYSTEM_PROMPT in prompt

    def test_instruction_is_in_prompt(self):
        instruction = "What is portfolio diversification?"
        prompt = format_for_inference({"instruction": instruction, "input": ""})
        assert instruction in prompt

    def test_input_context_included_when_provided(self):
        ex = {"instruction": "Calculate return.", "input": "Initial: $1000, Final: $1200"}
        prompt = format_for_inference(ex)
        assert "Initial: $1000" in prompt

    def test_output_not_leaked_into_inference_prompt(self):
        """The model answer must NEVER appear in the inference prompt."""
        ex = {"instruction": "Q?", "input": "", "output": "SECRET_ANSWER_XYZ"}
        prompt = format_for_inference(ex)
        assert "SECRET_ANSWER_XYZ" not in prompt

    def test_training_text_contains_output(self):
        ex = {
            "instruction": "What is inflation?",
            "input": "",
            "output": "Inflation is rising prices over time.",
        }
        full = format_for_training(ex)
        assert "Inflation is rising prices over time." in full

    def test_training_text_ends_with_im_end(self):
        ex = {"instruction": "Q?", "input": "", "output": "Answer text here."}
        full = format_for_training(ex)
        assert full.endswith("<|im_end|>")

    def test_training_text_longer_than_inference_prompt(self):
        ex = {"instruction": "Q?", "input": "", "output": "A long answer."}
        inference = format_for_inference(ex)
        training = format_for_training(ex)
        assert len(training) > len(inference)

    def test_prompt_is_deterministic(self):
        """Same input must always produce identical output."""
        ex = {"instruction": "Define bull market.", "input": ""}
        assert format_for_inference(ex) == format_for_inference(ex)

    def test_no_input_template_does_not_show_empty_input_section(self):
        """When input is empty, the prompt should not contain '### Input:\n\n'."""
        ex = {"instruction": "What is a bond?", "input": ""}
        prompt = format_for_inference(ex)
        # Should not have an empty ### Input: block
        assert "### Input:\n\n" not in prompt

    def test_with_input_shows_input_section(self):
        """When input is provided, ### Input: section should appear."""
        ex = {"instruction": "Calculate.", "input": "x=5, y=10"}
        prompt = format_for_inference(ex)
        assert "### Input:" in prompt
        assert "x=5, y=10" in prompt


# ---------------------------------------------------------------------------
# FormattedExample dataclass
# ---------------------------------------------------------------------------

class TestFormattedExample:
    def test_has_input_false_for_empty_input(self):
        ex = {"instruction": "Q?", "input": "", "output": "A."}
        fe = build_formatted_example(ex)
        assert fe.has_input is False

    def test_has_input_true_for_non_empty_input(self):
        ex = {"instruction": "Q?", "input": "Some context.", "output": "A."}
        fe = build_formatted_example(ex)
        assert fe.has_input is True

    def test_fields_correctly_populated(self):
        ex = {
            "instruction": "What is a mutual fund?",
            "input": "Focus on equity funds.",
            "output": "A mutual fund pools investor money.",
        }
        fe = build_formatted_example(ex)
        assert fe.instruction == ex["instruction"]
        assert fe.input == ex["input"]
        assert fe.output == ex["output"]
        assert ex["output"] in fe.full_text
        assert ex["output"] not in fe.prompt

    def test_prompt_matches_format_for_inference(self):
        ex = {"instruction": "What is ROI?", "input": "", "output": "Return on investment."}
        fe = build_formatted_example(ex)
        assert fe.prompt == format_for_inference(ex)

    def test_full_text_matches_format_for_training(self):
        ex = {"instruction": "What is ROI?", "input": "", "output": "Return on investment."}
        fe = build_formatted_example(ex)
        assert fe.full_text == format_for_training(ex)

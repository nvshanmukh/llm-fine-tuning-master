#!/usr/bin/env python3
"""
CLI inference script for the Finance LLM.

Supports single-shot query, interactive chat loop, and a
smoke-test mode that validates the full pipeline without
loading any model (useful for CI and CPU-only environments).

Usage:
    # Single query
    python scripts/inference.py --instruction "What is a mutual fund?"

    # With fine-tuned QLoRA adapter
    python scripts/inference.py \\
        --adapter-path ./experiments/qlora/final_model --4bit \\
        --instruction "Explain compound interest"

    # Interactive chat loop
    python scripts/inference.py --interactive

    # Smoke test (no model download, validates imports and logic)
    python scripts/inference.py --smoke-test
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer
from loguru import logger

from src.utils.config_utils import load_env_overrides
from src.utils.logging_utils import setup_logger

app = typer.Typer(help="CLI inference for the Finance LLM.")


@app.command()
def main(
    instruction: str = typer.Option(
        None,
        "--instruction",
        "-i",
        help="A financial question or instruction to answer.",
    ),
    input_context: str = typer.Option(
        "",
        help="Optional additional context to accompany the instruction.",
    ),
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-1.5B",
        help="Base model path or HuggingFace model ID.",
    ),
    adapter_path: str = typer.Option(
        None,
        help="Optional path to a LoRA/QLoRA adapter directory.",
    ),
    max_new_tokens: int = typer.Option(256, help="Maximum tokens to generate."),
    temperature: float = typer.Option(
        0.1,
        help="Sampling temperature (lower = more deterministic).",
    ),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling parameter."),
    load_in_4bit: bool = typer.Option(
        False,
        "--4bit",
        help="Load model in 4-bit quantization (for QLoRA inference).",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Start an interactive question-answering loop.",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Validate the pipeline without downloading or loading any model.",
    ),
    real_smoke_test: bool = typer.Option(
        False,
        "--real-smoke-test",
        help="Load a small real model and run one generation through FinanceLLMPredictor.",
    ),
    smoke_model: str = typer.Option(
        "sshleifer/tiny-gpt2",
        help="Small model id to use for --real-smoke-test (plumbing only).",
    ),
    log_level: str = typer.Option("INFO", help="Log level."),
) -> None:
    """Run inference with the Finance LLM."""
    setup_logger(level=log_level)
    load_env_overrides()

    if smoke_test:
        _run_smoke_test()
        return

    if real_smoke_test:
        _run_real_smoke_test(smoke_model)
        return

    from src.data.prompt_template import format_for_inference
    from src.inference.predict import FinanceLLMPredictor, GenerationConfig

    model_id = "base"
    if adapter_path:
        model_id = "qlora" if load_in_4bit else "lora"

    predictor = FinanceLLMPredictor(
        model_path=model_path,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        model_id=model_id,
    )

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0.0,
    )

    if interactive:
        _interactive_loop(predictor, gen_config)
    elif instruction:
        prompt = format_for_inference({"instruction": instruction, "input": input_context})
        result = predictor.generate(prompt, gen_config)
        print(f"\n{'='*60}")
        print(f"Model:   {result.model_id}")
        print(f"Latency: {result.latency_ms:.1f} ms  |  "
              f"Tokens: {result.input_tokens} in -> {result.output_tokens} out")
        print(f"{'='*60}")
        print(result.response)
        print(f"{'='*60}\n")
    else:
        logger.error("Provide --instruction or use --interactive or --smoke-test mode.")
        raise typer.Exit(code=1)


def _interactive_loop(predictor, gen_config) -> None:
    """Run an interactive financial Q&A loop with rich formatting."""
    from src.data.prompt_template import format_for_inference

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        console.print(Panel.fit(
            f"[bold green]Finance LLM -- Interactive Mode[/bold green]\n"
            f"Model: [cyan]{predictor.model_id}[/cyan]\n"
            f"Type [bold]'quit'[/bold] or [bold]'exit'[/bold] to stop.",
            title="Finance Q&A",
        ))

        while True:
            try:
                question = console.input("\n[bold yellow]> Question:[/bold yellow] ").strip()
                if question.lower() in {"quit", "exit", "q"}:
                    console.print("[dim]Goodbye![/dim]")
                    break
                if not question:
                    continue

                prompt = format_for_inference({"instruction": question, "input": ""})
                with console.status("[dim]Generating...[/dim]"):
                    result = predictor.generate(prompt, gen_config)

                console.print(Panel(
                    Text(result.response),
                    title=(
                        f"[cyan]{predictor.model_id}[/cyan] -- "
                        f"{result.latency_ms:.0f}ms | "
                        f"{result.output_tokens} tokens"
                    ),
                    border_style="blue",
                ))

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Interrupted. Goodbye![/dim]")
                break

    except ImportError:
        # Fallback: rich not installed
        print(f"\nFinance LLM Interactive Mode (Model: {predictor.model_id})")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                question = input("> Question: ").strip()
                if question.lower() in {"quit", "exit", "q"}:
                    print("Goodbye!")
                    break
                if not question:
                    continue
                prompt = format_for_inference({"instruction": question, "input": ""})
                result = predictor.generate(prompt, gen_config)
                print(f"\n[{predictor.model_id} | {result.latency_ms:.0f}ms]\n{result.response}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nInterrupted. Goodbye!")
                break


def _run_real_smoke_test(model_id: str) -> None:
    """
    Level-2 style check: exercise the *real* FinanceLLMPredictor code path
    (transformers load -> tokenize -> generate -> decode -> latency) with a
    small model. This proves the inference plumbing works end to end; it does
    NOT say anything about the quality of the target 1.5B model.
    """
    from src.data.prompt_template import format_for_inference
    from src.inference.predict import FinanceLLMPredictor, GenerationConfig

    logger.info(f"Real inference smoke test with model: {model_id}")
    predictor = FinanceLLMPredictor(model_path=model_id, model_id="real-smoke", device_map=None)
    predictor.warmup(GenerationConfig(max_new_tokens=4))

    prompt = format_for_inference({"instruction": "What is a mutual fund?", "input": ""})
    result = predictor.generate(prompt, GenerationConfig(max_new_tokens=16, do_sample=False))

    assert isinstance(result.response, str)
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms >= 0
    logger.info(
        f"[PASS] real inference smoke: {result.input_tokens} in -> "
        f"{result.output_tokens} out in {result.latency_ms:.0f} ms"
    )
    logger.info(f"model_size_mb={predictor.model_size_mb}")
    logger.info("=" * 60)
    logger.info("REAL INFERENCE SMOKE TEST PASSED (plumbing only, not model quality)")
    logger.info("=" * 60)


def _run_smoke_test() -> None:
    """
    Validate the full pipeline without loading any model.

    Tests:
    1. Prompt template formatting (with and without input context)
    2. Config loading from YAML
    3. Evaluation metrics (ROUGE, BLEU, exact match)
    4. Error analysis heuristics
    5. Inference dataclasses

    This test should pass on any machine, including CPU-only environments.
    """
    logger.info("Running smoke test -- no model download required...")

    # --- Test 1: Prompt template ---
    from src.data.prompt_template import (
        SYSTEM_PROMPT,
        build_formatted_example,
        format_for_inference,
        format_for_training,
    )

    ex_no_input = {
        "instruction": "What is inflation?",
        "input": "",
        "output": "Inflation is the rate of increase in prices over time.",
    }
    ex_with_input = {
        "instruction": "Calculate the return.",
        "input": "Initial: $100, Final: $120",
        "output": "The return is 20%.",
    }

    prompt = format_for_inference(ex_no_input)
    assert "What is inflation?" in prompt, "Instruction not in prompt"
    assert SYSTEM_PROMPT in prompt, "System prompt missing"
    assert "<|im_start|>assistant" in prompt, "Qwen ChatML format missing"
    assert ex_no_input["output"] not in prompt, "Output leaked into inference prompt"

    prompt_with_input = format_for_inference(ex_with_input)
    assert "Initial: $100" in prompt_with_input, "Input context not in prompt"

    full = format_for_training(ex_no_input)
    assert "Inflation is the rate" in full, "Output missing from training text"
    assert full.endswith("<|im_end|>"), "Training text does not end with im_end token"

    fe = build_formatted_example(ex_no_input)
    assert not fe.has_input, "has_input should be False for empty input"
    fe2 = build_formatted_example(ex_with_input)
    assert fe2.has_input, "has_input should be True for non-empty input"
    logger.info("  [PASS] Prompt template tests")

    # --- Test 2: Config loading ---
    from src.utils.config_utils import load_config
    cfg = load_config("configs/base.yaml")
    assert cfg.model.model_name_or_path == "Qwen/Qwen2.5-1.5B", "Model name mismatch"
    assert cfg.data.max_seq_length == 512, "Max seq length mismatch"
    logger.info("  [PASS] Config loading (base.yaml)")

    lora_cfg = load_config("configs/lora.yaml")
    assert not lora_cfg.model.load_in_4bit, "LoRA config should not use 4-bit"
    logger.info("  [PASS] Config loading (lora.yaml)")

    qlora_cfg = load_config("configs/qlora.yaml")
    assert qlora_cfg.model.load_in_4bit, "QLoRA config should use 4-bit"
    logger.info("  [PASS] Config loading (qlora.yaml)")

    # --- Test 3: Evaluation metrics ---
    from src.evaluation.metrics import (
        compute_bleu,
        compute_exact_match,
        compute_length_stats,
        compute_rouge,
    )

    preds = ["inflation is the rate of price increase"]
    refs  = ["inflation is the rate of price increase"]

    r = compute_rouge(preds, refs)
    assert r["rouge1"] > 0.9, f"ROUGE-1 should be near 1.0 for identical texts: {r}"
    assert r["rougeL"] > 0.9, f"ROUGE-L should be near 1.0 for identical texts: {r}"

    b = compute_bleu(preds, refs)
    assert b["bleu4"] > 0.0, f"BLEU-4 should be positive for identical texts: {b}"

    em = compute_exact_match(preds, refs)
    assert em["exact_match"] == 1.0, f"Exact match should be 1.0 for identical texts: {em}"

    em_partial = compute_exact_match(["correct", "wrong"], ["correct", "right"])
    assert em_partial["exact_match"] == 0.5, f"Expected 0.5, got {em_partial}"

    stats = compute_length_stats(preds, refs)
    assert "pred_len_mean" in stats and "ref_len_mean" in stats
    logger.info("  [PASS] Evaluation metrics (ROUGE, BLEU, EM, length stats)")

    # --- Test 4: Error analysis ---
    from src.evaluation.error_analysis import (
        ERROR_CATEGORIES,
        categorize_error_heuristic,
    )
    cat_short = categorize_error_heuristic("Too short", "longer reference answer here", "question")
    assert cat_short == "incomplete_answer", f"Expected incomplete_answer, got {cat_short}"

    good_resp = "This is a sufficiently detailed financial answer about inflation and monetary policy"
    good_ref  = "This is a detailed answer about inflation and monetary policy in modern economies"
    cat_ok = categorize_error_heuristic(good_resp, good_ref, "What is inflation?")
    assert cat_ok == "acceptable", f"Expected acceptable, got {cat_ok}"

    assert len(ERROR_CATEGORIES) == 8, "ERROR_CATEGORIES should have 8 entries"
    logger.info("  [PASS] Error analysis heuristics")

    # --- Test 5: Inference dataclasses ---
    from src.inference.predict import GenerationConfig, InferenceResult
    gc = GenerationConfig()
    assert gc.max_new_tokens == 256
    assert gc.do_sample is True

    ir = InferenceResult(
        prompt="test", response="answer", model_id="test",
        latency_ms=42.0, input_tokens=10, output_tokens=5,
    )
    assert ir.latency_ms == 42.0
    logger.info("  [PASS] Inference dataclasses")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED -- All pipeline components are functional.")
    logger.info("The project is ready for GPU-based training on a cloud environment.")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()

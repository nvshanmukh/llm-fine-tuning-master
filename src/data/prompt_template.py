"""
Prompt templating for the Finance Q&A fine-tuning task.

We use the Alpaca instruction-following format, adapted for the
Finance domain. This module provides functions to format raw
dataset examples into structured prompts for training and inference.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Alpaca-style template (used for both training and inference)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a knowledgeable financial advisor assistant. "
    "Answer questions accurately, concisely, and in plain English. "
    "Do not make up numbers or facts. If you are unsure, say so."
)

INSTRUCTION_TEMPLATE = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
### Instruction:
{instruction}

### Input:
{input}
<|im_end|>
<|im_start|>assistant
"""

RESPONSE_TEMPLATE = "{output}<|im_end|>"

TRAINING_TEMPLATE = INSTRUCTION_TEMPLATE + RESPONSE_TEMPLATE

INSTRUCTION_TEMPLATE_NO_INPUT = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
### Instruction:
{instruction}
<|im_end|>
<|im_start|>assistant
"""

TRAINING_TEMPLATE_NO_INPUT = INSTRUCTION_TEMPLATE_NO_INPUT + RESPONSE_TEMPLATE


@dataclass
class FormattedExample:
    """A fully formatted example ready for tokenization."""
    prompt: str                # The instruction/input portion (no answer)
    full_text: str             # Full text including the answer (for training)
    instruction: str           # Raw instruction
    input: str                 # Raw input (may be empty)
    output: str                # Raw expected output
    has_input: bool            # Whether the example has a non-empty input field


def format_for_training(example: dict) -> str:
    """
    Format a raw dataset example into a full training string.
    This includes both the prompt AND the expected answer.

    Args:
        example: Dict with keys: 'instruction', 'input', 'output'.

    Returns:
        Formatted string ready for causal LM training.
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if inp:
        text = TRAINING_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            input=inp,
            output=output,
        )
    else:
        text = TRAINING_TEMPLATE_NO_INPUT.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            output=output,
        )
    return text


def format_for_inference(example: dict) -> str:
    """
    Format a raw dataset example into an inference prompt.
    This does NOT include the expected output -- only the prompt.

    Args:
        example: Dict with keys: 'instruction', 'input'.

    Returns:
        Formatted prompt string ready for generation.
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()

    if inp:
        prompt = INSTRUCTION_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            input=inp,
        )
    else:
        prompt = INSTRUCTION_TEMPLATE_NO_INPUT.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
        )
    return prompt


def build_formatted_example(example: dict) -> FormattedExample:
    """
    Build a FormattedExample from a raw dataset dict.

    Args:
        example: Dict with 'instruction', 'input', 'output'.

    Returns:
        FormattedExample dataclass instance.
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()
    has_input = bool(inp)

    return FormattedExample(
        prompt=format_for_inference(example),
        full_text=format_for_training(example),
        instruction=instruction,
        input=inp,
        output=output,
        has_input=has_input,
    )

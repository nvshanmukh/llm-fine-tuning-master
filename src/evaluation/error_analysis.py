"""
Error analysis module for the Finance LLM fine-tuning project.

Categories failures into defined error types and produces structured
reports comparing Base vs Fine-tuned model outputs.

Error taxonomy:
1. hallucination         -- factual claims not in reference
2. incomplete_answer     -- response is cut off or too short
3. incorrect_reasoning   -- logical errors in explanation
4. instruction_failure   -- does not follow the instruction format
5. irrelevant_output     -- off-topic or completely wrong response
6. formatting_failure    -- wrong structure, bad formatting
7. domain_knowledge_gap  -- missing specific financial knowledge
8. acceptable            -- reasonable response, not a failure
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
ERROR_CATEGORIES = [
    "hallucination",
    "incomplete_answer",
    "incorrect_reasoning",
    "instruction_failure",
    "irrelevant_output",
    "formatting_failure",
    "domain_knowledge_gap",
    "acceptable",
]


@dataclass
class AnalyzedExample:
    """A single analyzed example with all model outputs and error categorization."""
    index: int
    instruction: str
    input: str
    reference: str
    base_output: str
    finetuned_output: str
    base_error_category: str = "acceptable"
    finetuned_error_category: str = "acceptable"
    base_rouge_l: float = 0.0
    finetuned_rouge_l: float = 0.0
    improvement: bool = False       # Did fine-tuning improve this example?
    regression: bool = False        # Did fine-tuning make this worse?
    analysis_notes: str = ""


REFUSAL_MARKERS = (
    "i'm not sure", "i am not sure", "i cannot", "i can't", "as an ai",
    "i do not have", "i don't have", "unable to provide", "i'm sorry",
)


def categorize_error_heuristic(response: str, reference: str, instruction: str) -> str:
    """
    Heuristically categorize a response into one of the defined error categories.

    This is a deterministic (no LLM, no human) FIRST-PASS categorization used to
    triage large evaluation sets. It is intentionally conservative and can only
    detect surface-level failure modes:

      - ``incomplete_answer``    -- very short output or explicit cut-off markers
      - ``irrelevant_output``    -- almost no lexical overlap with question/reference
      - ``domain_knowledge_gap`` -- explicit refusal / "I'm not sure" style answer
      - ``formatting_failure``   -- reference is clearly structured (list / numbered)
                                    but the response is a flat blob
      - ``instruction_failure``  -- response echoes the prompt scaffolding / template
      - ``hallucination``        -- heavy self-repetition (a weak proxy only)
      - ``acceptable``           -- passes all heuristic checks

    ``incorrect_reasoning`` is deliberately never assigned here: it cannot be
    detected without semantic understanding and must come from the LLM judge or
    human review pass.
    """
    resp = response.strip()
    resp_lower = resp.lower()
    response_words = set(resp_lower.split())
    reference_words = set(reference.lower().split())
    instruction_words = set(instruction.lower().split())

    # 1. Incomplete / truncated
    if len(resp.split()) < 10:
        return "incomplete_answer"
    if any(m in resp_lower for m in ("[cut", "[truncat")) or resp.endswith(("...", "-", ",")):
        return "incomplete_answer"

    # 2. Explicit refusal / knowledge gap
    if any(m in resp_lower for m in REFUSAL_MARKERS):
        return "domain_knowledge_gap"

    # 3. Leaked prompt scaffolding => did not follow the chat format
    if any(tok in resp for tok in ("<|im_start|>", "<|im_end|>", "### Instruction:", "### Input:")):
        return "instruction_failure"

    # 4. Irrelevant (minimal overlap with both reference and instruction)
    overlap_with_ref = len(response_words & reference_words) / max(len(reference_words), 1)
    overlap_with_inst = len(response_words & instruction_words) / max(len(instruction_words), 1)
    if overlap_with_ref < 0.05 and overlap_with_inst < 0.1:
        return "irrelevant_output"

    # 5. Repetition (weak hallucination proxy)
    sentences = [s.strip() for s in resp.split(".") if s.strip()]
    if len(sentences) > 3 and len(set(sentences)) / len(sentences) < 0.5:
        return "hallucination"

    # 6. Formatting: structured reference vs flat response
    ref_structured = bool(re.search(r"(^|\n)\s*(\d+[.)]|[-*])\s", reference))
    resp_structured = bool(re.search(r"(^|\n)\s*(\d+[.)]|[-*])\s", response))
    if ref_structured and not resp_structured and len(resp.split()) > 60:
        return "formatting_failure"

    return "acceptable"


def analyze_batch(
    examples: list[dict],
    base_outputs: list[str],
    finetuned_outputs: list[str],
) -> list[AnalyzedExample]:
    """
    Analyze a batch of examples by comparing base vs fine-tuned model outputs.

    Args:
        examples: List of raw dataset examples (instruction/input/output).
        base_outputs: Corresponding outputs from the base model.
        finetuned_outputs: Corresponding outputs from the fine-tuned model.

    Returns:
        List of AnalyzedExample objects.
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    analyzed = []
    for i, (ex, base_out, ft_out) in enumerate(zip(examples, base_outputs, finetuned_outputs)):
        reference = ex.get("output", "")
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")

        base_rouge = scorer.score(reference, base_out)["rougeL"].fmeasure
        ft_rouge = scorer.score(reference, ft_out)["rougeL"].fmeasure

        base_cat = categorize_error_heuristic(base_out, reference, instruction)
        ft_cat = categorize_error_heuristic(ft_out, reference, instruction)

        improvement = ft_rouge > base_rouge + 0.05  # significant improvement
        regression = base_rouge > ft_rouge + 0.05   # significant regression

        analyzed.append(AnalyzedExample(
            index=i,
            instruction=instruction,
            input=inp,
            reference=reference,
            base_output=base_out,
            finetuned_output=ft_out,
            base_error_category=base_cat,
            finetuned_error_category=ft_cat,
            base_rouge_l=round(base_rouge, 4),
            finetuned_rouge_l=round(ft_rouge, 4),
            improvement=improvement,
            regression=regression,
        ))

    logger.info(
        f"Error analysis: {sum(a.improvement for a in analyzed)} improvements, "
        f"{sum(a.regression for a in analyzed)} regressions, "
        f"{len(analyzed)} total"
    )
    return analyzed


def generate_error_report(
    analyzed: list[AnalyzedExample],
    output_path: Path | None = None,
) -> dict:
    """
    Generate a structured error analysis report.

    Args:
        analyzed: List of AnalyzedExample objects.
        output_path: Optional path to save the report as JSON.

    Returns:
        Dict with summary statistics and representative examples.
    """
    total = len(analyzed)
    if total == 0:
        return {"summary": {"total_examples": 0}, "error_distribution": {}, "representative_examples": {}}
    improvements = [a for a in analyzed if a.improvement]
    regressions = [a for a in analyzed if a.regression]

    # Compute error category distribution
    base_cat_counts: dict[str, int] = dict.fromkeys(ERROR_CATEGORIES, 0)
    ft_cat_counts: dict[str, int] = dict.fromkeys(ERROR_CATEGORIES, 0)
    for a in analyzed:
        base_cat_counts[a.base_error_category] = base_cat_counts.get(a.base_error_category, 0) + 1
        ft_cat_counts[a.finetuned_error_category] = ft_cat_counts.get(a.finetuned_error_category, 0) + 1

    # Representative examples (top 5 improvements, top 5 regressions, top 5 failures)
    top_improvements = sorted(improvements, key=lambda a: a.finetuned_rouge_l - a.base_rouge_l, reverse=True)[:5]
    top_regressions = sorted(regressions, key=lambda a: a.base_rouge_l - a.finetuned_rouge_l, reverse=True)[:5]
    persistent_failures = [
        a for a in analyzed
        if a.finetuned_error_category != "acceptable" and a.base_error_category != "acceptable"
    ][:5]

    def _example_to_dict(a: AnalyzedExample) -> dict:
        return {
            "index": a.index,
            "instruction": a.instruction[:200],
            "reference": a.reference[:300],
            "base_output": a.base_output[:300],
            "finetuned_output": a.finetuned_output[:300],
            "base_rouge_l": a.base_rouge_l,
            "finetuned_rouge_l": a.finetuned_rouge_l,
            "base_error": a.base_error_category,
            "finetuned_error": a.finetuned_error_category,
            "delta_rouge_l": round(a.finetuned_rouge_l - a.base_rouge_l, 4),
        }

    report = {
        "summary": {
            "total_examples": total,
            "improvements": len(improvements),
            "regressions": len(regressions),
            "neutral": total - len(improvements) - len(regressions),
            "improvement_rate": round(len(improvements) / total, 4),
            "regression_rate": round(len(regressions) / total, 4),
        },
        "error_distribution": {
            "base_model": base_cat_counts,
            "finetuned_model": ft_cat_counts,
        },
        "representative_examples": {
            "top_improvements": [_example_to_dict(a) for a in top_improvements],
            "top_regressions": [_example_to_dict(a) for a in top_regressions],
            "persistent_failures": [_example_to_dict(a) for a in persistent_failures],
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Error analysis report saved to: {output_path}")

    return report
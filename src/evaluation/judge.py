"""
LLM-as-a-Judge evaluation module.

Implements structured quality evaluation using an LLM (e.g. GPT-4 via API,
or a local model) to assess generated responses on multiple dimensions:
- Correctness: Is the factual content accurate?
- Relevance: Does the response address the question?
- Completeness: Is the answer sufficiently detailed?
- Instruction Following: Does the response follow the given instruction?
- Hallucination: Does the response contain fabricated information?

IMPORTANT LIMITATIONS (documented per project requirements):
1. LLM judges have their own biases (verbosity bias, positional bias).
2. LLM judge scores are NOT ground truth -- treat as one signal among many.
3. Without human annotation as calibration, scores are relative, not absolute.
4. This implementation should be combined with automatic metrics (ROUGE, BLEU, etc.).

This module is designed to work with any OpenAI-compatible API (including
local LM Studio, Ollama, or cloud providers).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from loguru import logger

# ---------------------------------------------------------------------------
# Judgment prompt template
# ---------------------------------------------------------------------------
JUDGMENT_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated financial advice.
You will be given a financial question, a reference answer, and a candidate answer.
Evaluate the candidate answer on the following dimensions using a 1-5 integer scale.

Do NOT favour longer answers simply because they are longer.
Be strict and honest. Return ONLY valid JSON."""

JUDGMENT_USER_TEMPLATE = """## Financial Question
{instruction}

## Input Context (if any)
{input}

## Reference Answer
{reference}

## Candidate Answer
{candidate}

---
Evaluate the CANDIDATE ANSWER on these dimensions (1=very poor, 5=excellent):

1. **correctness** (1-5): Is the factual content accurate and free from errors?
2. **relevance** (1-5): Does the response directly address the specific question asked?
3. **completeness** (1-5): Is the answer appropriately detailed and complete?
4. **instruction_following** (1-5): Does the response follow the instruction format and constraints?
5. **hallucination_score** (1-5): 5=no hallucination, 1=significant fabricated information.

Also provide a brief **reasoning** (2-3 sentences).

Return ONLY valid JSON in exactly this format:
{{
  "correctness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "instruction_following": <1-5>,
  "hallucination_score": <1-5>,
  "reasoning": "<string>",
  "overall": <1-5>
}}"""


@dataclass
class JudgmentResult:
    """Structured result from LLM-as-a-judge evaluation."""
    correctness: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    instruction_following: float = 0.0
    hallucination_score: float = 0.0
    overall: float = 0.0
    reasoning: str = ""
    raw_response: str = ""
    parsing_failed: bool = False


def parse_judgment(response_text: str) -> JudgmentResult:
    """
    Parse the LLM judge's JSON response into a JudgmentResult.

    Handles malformed JSON gracefully by extracting what it can.

    Args:
        response_text: Raw string response from the judge LLM.

    Returns:
        JudgmentResult with parsed scores.
    """
    # Try to extract JSON block from response
    json_match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
    if not json_match:
        logger.warning(f"Could not find JSON in judge response: {response_text[:100]}")
        return JudgmentResult(raw_response=response_text, parsing_failed=True)

    try:
        data = json.loads(json_match.group())
        return JudgmentResult(
            correctness=float(data.get("correctness", 0)),
            relevance=float(data.get("relevance", 0)),
            completeness=float(data.get("completeness", 0)),
            instruction_following=float(data.get("instruction_following", 0)),
            hallucination_score=float(data.get("hallucination_score", 0)),
            overall=float(data.get("overall", 0)),
            reasoning=data.get("reasoning", ""),
            raw_response=response_text,
            parsing_failed=False,
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parsing error: {e} | response: {response_text[:200]}")
        return JudgmentResult(raw_response=response_text, parsing_failed=True)


def judge_single(
    instruction: str,
    inp: str,
    reference: str,
    candidate: str,
    api_base_url: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> JudgmentResult:
    """
    Evaluate a single candidate response using an LLM judge.

    Args:
        instruction: The original instruction/question.
        inp: The input context (may be empty).
        reference: The ground-truth reference answer.
        candidate: The model-generated candidate answer.
        api_base_url: OpenAI-compatible API base URL.
        api_key: API key.
        model: Judge model name.
        temperature: Generation temperature (0.0 for deterministic).

    Returns:
        JudgmentResult with scores for each dimension.
    """
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base_url, api_key=api_key)

        user_message = JUDGMENT_USER_TEMPLATE.format(
            instruction=instruction,
            input=inp or "(no additional input)",
            reference=reference,
            candidate=candidate,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=512,
        )
        response_text = response.choices[0].message.content or ""
        return parse_judgment(response_text)

    except Exception as e:
        logger.error(f"LLM judge API call failed: {e}")
        return JudgmentResult(raw_response=str(e), parsing_failed=True)


def aggregate_judgments(results: list[JudgmentResult]) -> dict[str, float]:
    """
    Aggregate multiple JudgmentResults into mean scores.

    Excludes failed/unparseable results from the average.

    Args:
        results: List of JudgmentResult objects.

    Returns:
        Dict with mean scores for each dimension.
    """
    valid = [r for r in results if not r.parsing_failed]
    if not valid:
        logger.warning("No valid judgments to aggregate")
        return {}

    n = len(valid)
    return {
        "judge_correctness": round(sum(r.correctness for r in valid) / n, 3),
        "judge_relevance": round(sum(r.relevance for r in valid) / n, 3),
        "judge_completeness": round(sum(r.completeness for r in valid) / n, 3),
        "judge_instruction_following": round(sum(r.instruction_following for r in valid) / n, 3),
        "judge_hallucination_score": round(sum(r.hallucination_score for r in valid) / n, 3),
        "judge_overall": round(sum(r.overall for r in valid) / n, 3),
        "judge_valid_count": n,
        "judge_failed_count": len(results) - n,
    }
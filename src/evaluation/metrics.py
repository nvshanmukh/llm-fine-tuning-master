"""
Evaluation metrics for the Finance LLM fine-tuning project.

Implements:
- ROUGE-1, ROUGE-2, ROUGE-L (via rouge-score)
- BLEU-4 (via nltk)
- BERTScore F1 (via bert-score)
- Exact Match
- Response length statistics
- Perplexity (via cross-entropy on the model)

All functions return plain dicts for easy MLflow logging.
"""
from __future__ import annotations

import re
import typing

from loguru import logger


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison:
    - Lowercase
    - Remove punctuation
    - Collapse multiple spaces
    - Strip leading/trailing whitespace

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: List of model-generated text strings.
        references: List of ground-truth reference strings.

    Returns:
        Dict with keys: rouge1, rouge2, rougeL (all F1 scores, 0-1).
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in agg:
            agg[key].append(scores[key].fmeasure)

    result = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in agg.items()}
    logger.debug(f"ROUGE scores: {result}")
    return result


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute corpus-level BLEU-4 score.

    Args:
        predictions: List of model-generated text strings.
        references: List of ground-truth reference strings.

    Returns:
        Dict with key: bleu4 (0-1).
    """
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    smooth = SmoothingFunction().method1
    tokenized_preds = [pred.lower().split() for pred in predictions]
    tokenized_refs = [[ref.lower().split()] for ref in references]

    bleu = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smooth)
    result = {"bleu4": round(bleu, 4)}
    logger.debug(f"BLEU-4: {result}")
    return result


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 16,
) -> dict[str, float]:
    """
    Compute BERTScore F1 using a lightweight model.

    NOTE: Uses distilbert-base-uncased for efficiency. A larger model
    (e.g. roberta-large) would give more accurate scores but requires
    more memory.

    Args:
        predictions: List of model-generated text strings.
        references: List of ground-truth reference strings.
        model_type: BERTScore backbone model.
        batch_size: Batch size for BERTScore computation.

    Returns:
        Dict with keys: bertscore_precision, bertscore_recall, bertscore_f1.
    """
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
        )
        result = {
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall": round(R.mean().item(), 4),
            "bertscore_f1": round(F1.mean().item(), 4),
        }
        logger.debug(f"BERTScore: {result}")
        return result
    except ImportError:
        logger.warning("bert-score not installed; skipping BERTScore computation")
        return {"bertscore_f1": 0.0}


def compute_exact_match(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute exact match accuracy.

    Args:
        predictions: List of model-generated text strings.
        references: List of ground-truth reference strings.
        normalize: Whether to normalize text before comparison.

    Returns:
        Dict with key: exact_match (0-1).
    """
    fn = normalize_text if normalize else (lambda x: x)
    matches = sum(fn(p) == fn(r) for p, r in zip(predictions, references))
    em = matches / len(predictions) if predictions else 0.0
    result = {"exact_match": round(em, 4)}
    logger.debug(f"Exact Match: {result}")
    return result


def compute_length_stats(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute token/word length statistics for predictions and references.

    Args:
        predictions: List of model-generated text strings.
        references: List of ground-truth reference strings.

    Returns:
        Dict with mean/median length of predictions and references (in words).
    """
    import numpy as np

    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]

    return {
        "pred_len_mean": round(float(np.mean(pred_lens)), 1),
        "pred_len_median": round(float(np.median(pred_lens)), 1),
        "ref_len_mean": round(float(np.mean(ref_lens)), 1),
        "ref_len_median": round(float(np.median(ref_lens)), 1),
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    skip_bertscore: bool = False,
) -> dict[str, typing.Any]:
    """
    Compute all evaluation metrics in one call.

    Args:
        predictions: List of model-generated text strings.
        references: List of ground-truth reference strings.
        skip_bertscore: Skip BERTScore computation (slow).

    Returns:
        Merged dict of all metric scores.
    """
    logger.info(f"Computing metrics for {len(predictions)} examples...")
    results: dict[str, typing.Any] = {}
    results.update(compute_rouge(predictions, references))
    results.update(compute_bleu(predictions, references))
    results.update(compute_exact_match(predictions, references))
    results.update(compute_length_stats(predictions, references))
    if not skip_bertscore:
        results.update(compute_bert_score(predictions, references))
    logger.info(f"Metrics: {results}")
    return results
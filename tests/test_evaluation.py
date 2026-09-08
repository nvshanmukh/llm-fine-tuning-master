"""
Tests for the evaluation metrics and error analysis modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.evaluation.error_analysis import (
    ERROR_CATEGORIES,
    analyze_batch,
    categorize_error_heuristic,
    generate_error_report,
)
from src.evaluation.judge import judge_pairwise, parse_judgment
from src.evaluation.metrics import (
    bootstrap_mean_ci,
    compute_all_metrics,
    compute_bleu,
    compute_exact_match,
    compute_length_stats,
    compute_rouge,
    normalize_text,
    paired_bootstrap_diff,
    rouge_l_per_example,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("HELLO") == "hello"

    def test_punctuation_removed(self):
        result = normalize_text("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_extra_spaces_collapsed(self):
        result = normalize_text("hello   world")
        assert "  " not in result

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestROUGE:
    def test_perfect_match(self):
        scores = compute_rouge(["hello world"], ["hello world"])
        assert scores["rouge1"] == pytest.approx(1.0, abs=0.01)
        assert scores["rougeL"] == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        scores = compute_rouge(["aaa bbb ccc"], ["xxx yyy zzz"])
        assert scores["rouge1"] < 0.1

    def test_partial_overlap(self):
        scores = compute_rouge(["the cat sat on the mat"], ["the cat sat"])
        assert 0.0 < scores["rouge1"] < 1.0

    def test_batch_returns_average(self):
        scores = compute_rouge(
            ["perfect match", "no overlap at all"],
            ["perfect match", "completely different words"],
        )
        # Average of 1.0 and ~0.0 should be ~0.5
        assert 0.2 < scores["rouge1"] < 0.8

    def test_returns_correct_keys(self):
        scores = compute_rouge(["test"], ["test"])
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores


class TestBLEU:
    def test_perfect_match(self):
        scores = compute_bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
        # BLEU-4 with smoothing on a short perfect match
        assert scores["bleu4"] > 0.5

    def test_no_overlap_gives_low_bleu(self):
        scores = compute_bleu(["totally different words here"], ["nothing matches at all"])
        assert scores["bleu4"] < 0.3

    def test_returns_correct_keys(self):
        scores = compute_bleu(["test"], ["test"])
        assert "bleu4" in scores


class TestExactMatch:
    def test_exact_match_normalized(self):
        em = compute_exact_match(["Inflation Is Rising!"], ["inflation is rising"], normalize=True)
        assert em["exact_match"] == pytest.approx(1.0)

    def test_exact_match_unnormalized_mismatch(self):
        em = compute_exact_match(["Inflation"], ["inflation"], normalize=False)
        assert em["exact_match"] == pytest.approx(0.0)

    def test_partial_match_gives_partial_score(self):
        em = compute_exact_match(["correct", "wrong"], ["correct", "right"])
        assert em["exact_match"] == pytest.approx(0.5)

    def test_empty_list(self):
        em = compute_exact_match([], [])
        assert em["exact_match"] == 0.0


class TestLengthStats:
    def test_returns_all_keys(self):
        stats = compute_length_stats(["hello world"], ["hello world today"])
        assert "pred_len_mean" in stats
        assert "ref_len_mean" in stats

    def test_correct_lengths(self):
        stats = compute_length_stats(["one two three"], ["one two"])
        assert stats["pred_len_mean"] == pytest.approx(3.0)
        assert stats["ref_len_mean"] == pytest.approx(2.0)


class TestComputeAllMetrics:
    def test_returns_all_expected_metrics(self):
        preds = ["the economy grew steadily"]
        refs = ["the economy grew steadily"]
        metrics = compute_all_metrics(preds, refs, skip_bertscore=True)
        assert "rouge1" in metrics
        assert "rouge2" in metrics
        assert "rougeL" in metrics
        assert "bleu4" in metrics
        assert "exact_match" in metrics


class TestBootstrap:
    def test_rouge_l_per_example_length(self):
        scores = rouge_l_per_example(["a b c", "x y"], ["a b c", "p q"])
        assert len(scores) == 2
        assert scores[0] == pytest.approx(1.0, abs=0.01)

    def test_ci_brackets_mean(self):
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci = bootstrap_mean_ci(vals, n_resamples=500, seed=1)
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
        assert ci["n"] == 5

    def test_ci_empty(self):
        ci = bootstrap_mean_ci([], n_resamples=100)
        assert ci == {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    def test_paired_diff_detects_improvement(self):
        a = [0.8, 0.9, 0.7, 0.85, 0.95]
        b = [0.2, 0.3, 0.1, 0.25, 0.15]
        d = paired_bootstrap_diff(a, b, n_resamples=500, seed=1)
        assert d["mean_diff"] > 0.4
        assert d["prob_a_gt_b"] == pytest.approx(1.0, abs=0.01)

    def test_paired_diff_no_difference(self):
        a = [0.5, 0.5, 0.5, 0.5]
        d = paired_bootstrap_diff(a, list(a), n_resamples=200, seed=1)
        assert d["mean_diff"] == pytest.approx(0.0)


class TestJudge:
    def test_parse_valid_json(self):
        r = parse_judgment('{"correctness": 4, "relevance": 5, "completeness": 3, '
                            '"instruction_following": 4, "hallucination_score": 5, '
                            '"reasoning": "ok", "overall": 4}')
        assert not r.parsing_failed
        assert r.correctness == 4.0
        assert r.overall == 4.0

    def test_parse_garbage(self):
        r = parse_judgment("the model did fine, no json here")
        assert r.parsing_failed

    def test_pairwise_blinding_and_swap_mapping(self):
        # Judge always says the first-presented answer ("A") wins.
        calls = []

        def fake_llm(system, user):
            calls.append(user)
            return '{"winner": "A", "reasoning": "x"}'

        # no swap: A == answer_a -> winner "a"
        r1 = judge_pairwise("q", "", "ref", "ANS_A", "ANS_B", fake_llm, swap=False)
        assert r1["winner"] == "a"
        # swap: presented order reversed, "A" maps back to answer_b
        r2 = judge_pairwise("q", "", "ref", "ANS_A", "ANS_B", fake_llm, swap=True)
        assert r2["winner"] == "b"
        # identity of the answers is in the prompt but not their labels
        assert "ANS_A" in calls[0] and "ANS_B" in calls[0]


class TestErrorAnalysis:
    def test_incomplete_answer_short(self):
        cat = categorize_error_heuristic("Too short", "reference answer is much longer", "question")
        assert cat == "incomplete_answer"

    def test_acceptable_answer(self):
        good = "This is a good and detailed answer about financial instruments and markets"
        ref = "This is a detailed answer about financial instruments and markets"
        cat = categorize_error_heuristic(good, ref, "financial question")
        assert cat == "acceptable"

    def test_analyze_batch_returns_correct_length(self):
        examples = [
            {"instruction": "What is X?", "input": "", "output": "X is a financial instrument used for hedging risk"},
            {"instruction": "What is Y?", "input": "", "output": "Y is an investment vehicle that pools capital"},
        ]
        base_outputs = [
            "X is a financial instrument used for hedging risk",
            "I don't know",  # short
        ]
        ft_outputs = [
            "X is a financial instrument used for hedging risk and portfolio management",
            "Y is an investment vehicle that pools capital from multiple investors",
        ]
        analyzed = analyze_batch(examples, base_outputs, ft_outputs)
        assert len(analyzed) == 2

    def test_generate_error_report_structure(self):
        examples = [
            {"instruction": f"Q{i}", "input": "", "output": f"Answer {i} with sufficient length for analysis"}
            for i in range(10)
        ]
        base_outs = [ex["output"] for ex in examples]
        ft_outs = [ex["output"] for ex in examples]
        analyzed = analyze_batch(examples, base_outs, ft_outs)
        report = generate_error_report(analyzed)
        assert "summary" in report
        assert "error_distribution" in report
        assert "representative_examples" in report
        assert report["summary"]["total_examples"] == 10

    def test_error_categories_are_valid(self):
        for ex_type in ERROR_CATEGORIES:
            assert isinstance(ex_type, str)

"""
test_custom_metrics.py — LLM evaluation without RAGAS.

These tests demonstrate how to build lightweight evaluation metrics
using only pytest — no external eval framework needed.

Perfect for:
- Teams that can't afford LLM API calls in every CI run
- Deterministic checks that should never be flaky
- Fast feedback loops during development

Run:
    pytest tests/test_custom_metrics.py -v
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Metrics module tests
# ---------------------------------------------------------------------------


class TestResponseLength:
    def test_normal_answer_passes(self):
        from llm_eval.metrics import response_length_ok

        answer = (
            "Python generators use yield to produce values lazily "
            "without storing the full sequence in memory."
        )
        result = response_length_ok(answer, min_words=5, max_words=100)
        assert result.passed
        assert result.score == 1.0

    def test_too_short_fails(self):
        from llm_eval.metrics import response_length_ok

        result = response_length_ok("Yes.", min_words=10)
        assert not result.passed
        assert result.score == 0.0

    def test_too_long_fails(self):
        from llm_eval.metrics import response_length_ok

        long_answer = " ".join(["word"] * 600)
        result = response_length_ok(long_answer, max_words=500)
        assert not result.passed


class TestNoExplicitRefusal:
    @pytest.mark.parametrize("bad_answer", [
        "I don't know the answer to this question.",
        "I'm not sure about that.",
        "I cannot answer this.",
        "There's not enough information in the context.",
    ])
    def test_refusal_phrases_fail(self, bad_answer):
        from llm_eval.metrics import no_explicit_refusal

        result = no_explicit_refusal(bad_answer)
        assert not result.passed

    def test_good_answer_passes(self):
        from llm_eval.metrics import no_explicit_refusal

        answer = "The GIL prevents multiple threads from executing Python bytecode simultaneously."
        result = no_explicit_refusal(answer)
        assert result.passed


class TestKeywordCoverage:
    def test_all_keywords_present(self):
        from llm_eval.metrics import keyword_coverage

        answer = "The GIL is a mutex that protects Python objects from concurrent access."
        result = keyword_coverage(answer, keywords=["GIL", "mutex", "Python"])
        assert result.passed
        assert result.score == 1.0

    def test_partial_coverage_with_threshold(self):
        from llm_eval.metrics import keyword_coverage

        answer = "The GIL is a lock."
        result = keyword_coverage(
            answer, keywords=["GIL", "mutex", "Python", "thread"], threshold=0.3
        )
        # "GIL" is present → 1/4 = 0.25 < 0.3 → fails
        assert result.score == pytest.approx(0.25)

    def test_threshold_50_percent(self):
        from llm_eval.metrics import keyword_coverage

        answer = "Generators use yield and are memory efficient."
        result = keyword_coverage(
            answer, keywords=["yield", "memory", "thread", "async"], threshold=0.5
        )
        # yield, memory present → 2/4 = 0.5 ≥ 0.5 → passes
        assert result.passed


class TestContextGrounding:
    def test_well_grounded_answer(self):
        from llm_eval.metrics import context_grounding

        context = ["The GIL is a mutex that prevents parallel execution of Python bytecode."]
        answer = "The GIL is a mutex preventing parallel Python execution."
        result = context_grounding(answer, context, threshold=0.3)
        assert result.passed

    def test_unrelated_answer_low_score(self):
        from llm_eval.metrics import context_grounding

        context = ["Generators use yield to produce values lazily."]
        answer = "Quantum computing uses qubits for exponential speedup."
        result = context_grounding(answer, context, threshold=0.5)
        assert not result.passed


class TestOverallScore:
    def test_overall_score_all_pass(self):
        from llm_eval.metrics import MetricResult, overall_score

        results = [
            MetricResult("a", 0.9, True),
            MetricResult("b", 0.8, True),
            MetricResult("c", 1.0, True),
        ]
        assert overall_score(results) == pytest.approx(0.9)

    def test_overall_score_empty(self):
        from llm_eval.metrics import overall_score

        assert overall_score([]) == 0.0


# ---------------------------------------------------------------------------
# Dataset validation tests
# ---------------------------------------------------------------------------


class TestDatasetQuality:
    """Make sure the sample dataset itself is well-formed."""

    def test_all_required_fields_present(self, sample_qa_data):
        required = {"question", "contexts", "answer", "ground_truth"}
        for i, item in enumerate(sample_qa_data):
            missing = required - item.keys()
            assert not missing, f"Row {i} missing fields: {missing}"

    def test_no_duplicate_questions(self, sample_qa_data):
        questions = [item["question"] for item in sample_qa_data]
        assert len(questions) == len(set(questions)), "Duplicate questions found in dataset"

    def test_ground_truth_different_from_answer(self, sample_qa_data):
        """Ground truth and model answer should not be identical (would make eval trivial)."""
        for item in sample_qa_data:
            assert item["answer"] != item["ground_truth"], (
                f"Answer identical to ground truth for: {item['question'][:50]}"
            )

    def test_minimum_dataset_size(self, sample_qa_data):
        assert len(sample_qa_data) >= 5, "Dataset should have at least 5 items for meaningful eval"

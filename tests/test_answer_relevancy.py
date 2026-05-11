"""
test_answer_relevancy.py — verify that answers actually address the question.

Answer Relevancy = "Did the model stay on topic?"

Analogy: If someone asks 'What time does the library open?'
         and the answer is about Roman history — that's low relevancy.

Run:
    pytest tests/test_answer_relevancy.py -v
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fast tests — no API key required
# ---------------------------------------------------------------------------


def test_answers_are_substantial(small_qa_data):
    """
    Answers should be at least 10 words long.

    Very short answers are often evasive or incomplete.
    """
    from llm_eval.metrics import response_length_ok

    failures = []
    for item in small_qa_data:
        result = response_length_ok(item["answer"], min_words=10, max_words=500)
        if not result.passed:
            failures.append(f"Q: {item['question'][:50]} | {result}")

    assert not failures, "Some answers failed length check:\n" + "\n".join(failures)


def test_answers_dont_refuse(small_qa_data):
    """Answers should not contain explicit refusal phrases like 'I don't know'."""
    from llm_eval.metrics import no_explicit_refusal

    failures = []
    for item in small_qa_data:
        result = no_explicit_refusal(item["answer"])
        if not result.passed:
            failures.append(f"Q: {item['question'][:50]} | {result}")

    assert not failures, "Some answers contain refusal phrases:\n" + "\n".join(failures)


@pytest.mark.parametrize("item_index,expected_keyword", [
    (0, "GIL"),
    (1, "fixture"),
    (2, "runtime"),
    (3, "yield"),
    (4, "RAGAS"),
])
def test_answer_contains_key_term(sample_qa_data, item_index, expected_keyword):
    """Spot-check that specific answers contain domain-relevant keywords."""
    answer = sample_qa_data[item_index]["answer"]
    assert expected_keyword.lower() in answer.lower(), (
        f"Expected '{expected_keyword}' in answer: '{answer[:100]}'"
    )


# ---------------------------------------------------------------------------
# RAGAS tests — require OPENAI_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.llm
def test_answer_relevancy_above_threshold(ragas_dataset, ragas_llm):
    """
    RAGAS answer relevancy score must be >= 0.7.

    RAGAS measures this by asking the LLM to generate synthetic questions
    from the answer and checking if they match the original question.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy
    except ImportError:
        pytest.skip("ragas not installed")

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[answer_relevancy],
        llm=ragas_llm,
    )
    score = result["answer_relevancy"]
    assert score >= 0.7, (
        f"Answer relevancy too low: {score:.2f} (threshold: 0.7)\n"
        f"Answers are not sufficiently addressing the questions."
    )

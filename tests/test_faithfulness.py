"""
test_faithfulness.py — verify that RAG answers are grounded in retrieved context.

Faithfulness = "Did the model only say things the context supports?"

Analogy: Like checking if a student's essay only uses information
         from the given reading materials — no making stuff up!

Run:
    pytest tests/test_faithfulness.py -v
    pytest tests/test_faithfulness.py -v -m ragas   # RAGAS tests only (need API key)
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fast tests — no API key required
# ---------------------------------------------------------------------------


def test_faithfulness_metric_import():
    """RAGAS faithfulness metric should be importable."""
    try:
        from ragas.metrics import faithfulness  # noqa: F401
    except ImportError:
        pytest.skip("ragas not installed")


def test_context_grounding_heuristic(small_qa_data):
    """
    Heuristic faithfulness check using our custom metric.

    This runs without any API calls — great for CI pipelines.
    """
    from llm_eval.metrics import context_grounding

    failures = []
    for item in small_qa_data:
        result = context_grounding(
            answer=item["answer"],
            contexts=item["contexts"],
            threshold=0.3,
        )
        if not result.passed:
            failures.append(f"Q: {item['question'][:50]} | {result}")

    assert not failures, "Some answers failed context grounding:\n" + "\n".join(failures)


def test_answer_not_empty(small_qa_data):
    """All answers in the dataset must be non-empty."""
    for item in small_qa_data:
        assert item["answer"].strip(), f"Empty answer for question: {item['question']}"


def test_contexts_not_empty(small_qa_data):
    """All dataset rows must have at least one non-empty context."""
    for item in small_qa_data:
        assert item["contexts"], f"No contexts for: {item['question']}"
        assert all(c.strip() for c in item["contexts"]), "Context list contains empty strings"


# ---------------------------------------------------------------------------
# RAGAS tests — require OPENAI_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.llm
def test_faithfulness_above_threshold(ragas_dataset, ragas_llm):
    """
    RAGAS faithfulness score must be >= 0.7.

    This is the canonical way to gate RAG quality in CI:
    fail the build if the model starts hallucinating.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
    except ImportError:
        pytest.skip("ragas not installed")

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
    )
    score = result["faithfulness"]
    assert score >= 0.7, (
        f"Faithfulness too low: {score:.2f} (threshold: 0.7)\n"
        f"The model is generating answers not supported by retrieved context."
    )


@pytest.mark.ragas
@pytest.mark.llm
def test_faithfulness_per_sample(ragas_dataset, ragas_llm):
    """
    Check faithfulness row-by-row to identify which questions cause hallucination.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from datasets import Dataset
    except ImportError:
        pytest.skip("ragas or datasets not installed")

    # Evaluate row by row for granular feedback
    failing = []
    for i in range(len(ragas_dataset)):
        row_dataset = Dataset.from_dict({
            "question": [ragas_dataset[i]["question"]],
            "answer": [ragas_dataset[i]["answer"]],
            "contexts": [ragas_dataset[i]["contexts"]],
            "ground_truth": [ragas_dataset[i]["ground_truth"]],
        })
        result = evaluate(dataset=row_dataset, metrics=[faithfulness], llm=ragas_llm)
        score = result["faithfulness"]
        if score < 0.7:
            failing.append(f"Row {i}: score={score:.2f}, Q={ragas_dataset[i]['question'][:60]}")

    assert not failing, "Low-faithfulness samples found:\n" + "\n".join(failing)

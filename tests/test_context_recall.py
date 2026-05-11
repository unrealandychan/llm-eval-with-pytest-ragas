"""
test_context_recall.py — verify that retrieved context covers the ground truth.

Context Recall = "Did we retrieve the right information?"

Analogy: If the correct answer requires knowing facts A, B, and C,
         but we only retrieved a document containing fact A —
         our context recall is 1/3 = 33%.

Run:
    pytest tests/test_context_recall.py -v
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fast tests — no API key required
# ---------------------------------------------------------------------------


def test_contexts_cover_ground_truth_keywords(small_qa_data):
    """
    Heuristic: check that the combined context contains key terms from ground truth.

    Doesn't need an LLM — quick CI-friendly check.
    """
    failures = []
    for item in small_qa_data:
        ground_truth_words = set(item["ground_truth"].lower().split())
        # Remove stop words
        stopwords = {"the", "a", "an", "is", "in", "it", "of", "to", "and", "or", "for", "with",
                     "are", "be", "by", "at", "as", "its"}
        ground_truth_words -= stopwords

        context_text = " ".join(item["contexts"]).lower()
        covered = sum(1 for w in ground_truth_words if w in context_text)
        recall = covered / len(ground_truth_words) if ground_truth_words else 1.0

        if recall < 0.4:
            failures.append(
                f"Q: {item['question'][:50]}\n"
                f"  Context recall (heuristic): {recall:.0%}"
            )

    assert not failures, "Low context recall detected:\n" + "\n".join(failures)


def test_each_item_has_context(small_qa_data):
    """Every Q&A pair must have at least one context chunk."""
    for item in small_qa_data:
        assert len(item["contexts"]) >= 1, f"No context for: {item['question']}"


def test_context_length_reasonable(small_qa_data):
    """Context chunks should be non-trivially long (at least 20 words)."""
    for item in small_qa_data:
        for ctx in item["contexts"]:
            word_count = len(ctx.split())
            assert word_count >= 20, (
                f"Context too short ({word_count} words) for: {item['question'][:50]}\n"
                f"Context: {ctx[:80]}"
            )


# ---------------------------------------------------------------------------
# RAGAS tests — require OPENAI_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.ragas
@pytest.mark.llm
def test_context_recall_above_threshold(ragas_dataset, ragas_llm):
    """
    RAGAS context recall score must be >= 0.7.

    Low context recall means your retriever isn't fetching the right documents.
    Fix: improve embeddings, chunking strategy, or use a re-ranker.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import context_recall
    except ImportError:
        pytest.skip("ragas not installed")

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[context_recall],
        llm=ragas_llm,
    )
    score = result["context_recall"]
    assert score >= 0.7, (
        f"Context recall too low: {score:.2f} (threshold: 0.7)\n"
        f"The retriever is not fetching documents that cover the ground truth."
    )


@pytest.mark.ragas
@pytest.mark.llm
def test_all_ragas_metrics(ragas_dataset, ragas_llm):
    """
    Run all three core RAGAS metrics in a single evaluation call.

    This is the most efficient way to do a full RAGAS eval — one API call
    batch instead of three separate ones.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
    except ImportError:
        pytest.skip("ragas not installed")

    THRESHOLDS = {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_recall": 0.7,
    }

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=ragas_llm,
    )

    failures = []
    for metric, threshold in THRESHOLDS.items():
        score = result[metric]
        if score < threshold:
            failures.append(f"{metric}: {score:.2f} < {threshold}")

    assert not failures, "RAGAS metrics below threshold:\n" + "\n".join(failures)

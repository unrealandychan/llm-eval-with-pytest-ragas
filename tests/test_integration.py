"""
test_integration.py — end-to-end pipeline tests.

These tests exercise the full RAG pipeline: retrieval → generation → evaluation.

Run (mock mode, no API key):
    pytest tests/test_integration.py -v

Run (real LLM):
    OPENAI_API_KEY=sk-... pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_rag_pipeline_returns_result(rag_pipeline):
    """RAG pipeline should return a RAGResult with all required fields."""
    from llm_eval.rag_pipeline import RAGResult

    result = rag_pipeline.run("What is the Python GIL?")

    assert isinstance(result, RAGResult)
    assert result.question == "What is the Python GIL?"
    assert result.answer  # non-empty
    assert isinstance(result.contexts, list)
    assert len(result.contexts) >= 1


@pytest.mark.integration
def test_rag_pipeline_retrieves_relevant_context(rag_pipeline):
    """The retriever should fetch context relevant to the question."""
    result = rag_pipeline.run("How do pytest fixtures work?")

    # At least one context should mention "fixture" or "pytest"
    combined = " ".join(result.contexts).lower()
    assert "fixture" in combined or "pytest" in combined, (
        f"Retrieved context doesn't seem relevant to pytest fixtures.\n"
        f"Contexts: {result.contexts}"
    )


@pytest.mark.integration
def test_rag_pipeline_batch(rag_pipeline, small_qa_data):
    """Batch processing should return one result per input."""
    results = rag_pipeline.run_batch(small_qa_data)
    assert len(results) == len(small_qa_data)


@pytest.mark.integration
def test_rag_pipeline_answer_quality(rag_pipeline, small_qa_data):
    """Run all custom metrics on pipeline outputs and check overall quality."""
    from llm_eval.metrics import run_all_metrics, overall_score

    results = rag_pipeline.run_batch(small_qa_data)
    all_scores = []

    for rag_result in results:
        metrics = run_all_metrics(
            answer=rag_result.answer,
            contexts=rag_result.contexts,
        )
        score = overall_score(metrics)
        all_scores.append(score)

    avg_score = sum(all_scores) / len(all_scores)
    assert avg_score >= 0.5, (
        f"Average quality score too low: {avg_score:.2f}\n"
        f"Per-question scores: {[f'{s:.2f}' for s in all_scores]}"
    )


@pytest.mark.integration
@pytest.mark.llm
def test_rag_pipeline_with_real_llm(real_llm_client):
    """
    End-to-end test with a real LLM.

    Skipped automatically if no API key is configured.
    """
    from llm_eval.rag_pipeline import RAGPipeline, SimpleRetriever
    from llm_eval.metrics import context_grounding

    pipeline = RAGPipeline(retriever=SimpleRetriever())
    pipeline.llm = real_llm_client

    result = pipeline.run(
        "What is the Python GIL?",
        ground_truth="The GIL prevents multiple threads from executing Python bytecode simultaneously.",
    )

    grounding = context_grounding(result.answer, result.contexts, threshold=0.25)
    assert grounding.passed, (
        f"Real LLM answer not grounded in context.\n"
        f"Answer: {result.answer}\n"
        f"Metric: {grounding}"
    )


@pytest.mark.integration
def test_retriever_returns_top_k(rag_pipeline):
    """SimpleRetriever should respect the top_k parameter."""
    from llm_eval.rag_pipeline import SimpleRetriever

    retriever = SimpleRetriever()
    docs = retriever.retrieve("Python GIL threading", top_k=1)
    assert len(docs) == 1

    docs = retriever.retrieve("Python GIL threading", top_k=3)
    assert len(docs) == 3


@pytest.mark.integration
def test_client_factory_mock_mode(monkeypatch):
    """get_client() should return MockLLMClient when LLM_EVAL_MODE=mock."""
    from llm_eval.client import get_client, MockLLMClient

    monkeypatch.setenv("LLM_EVAL_MODE", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    client = get_client()
    assert isinstance(client, MockLLMClient)


@pytest.mark.integration
def test_mock_client_returns_response():
    """MockLLMClient should return a predictable response."""
    from llm_eval.client import MockLLMClient, Message

    client = MockLLMClient()
    response = client.chat([Message(role="user", content="What is Python?")])

    assert response.content.startswith("[MOCK]")
    assert response.model == "mock-gpt"
    assert response.provider == "mock"

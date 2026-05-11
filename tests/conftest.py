"""
conftest.py — shared pytest fixtures for llm-eval-with-pytest-ragas.

Run ALL tests (needs API key):
    pytest

Run only fast mock tests (no API key needed):
    pytest -m "not llm and not ragas and not integration"

Run RAGAS tests only:
    pytest -m ragas
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# ---------------------------------------------------------------------------
# Custom pytest marks (registered in pyproject.toml)
# ---------------------------------------------------------------------------
# @pytest.mark.llm        — calls a real LLM API
# @pytest.mark.ragas      — uses the RAGAS evaluation framework
# @pytest.mark.integration — end-to-end pipeline tests


# ---------------------------------------------------------------------------
# Fixtures: raw data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_qa_data() -> list[dict[str, Any]]:
    """Load the sample Q&A dataset from datasets/sample_qa.json."""
    path = DATASETS_DIR / "sample_qa.json"
    with path.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def small_qa_data(sample_qa_data) -> list[dict[str, Any]]:
    """First 3 items only — for fast tests."""
    return sample_qa_data[:3]


# ---------------------------------------------------------------------------
# Fixtures: LLM client
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llm_client():
    """
    Return an LLM client.

    - If LLM_EVAL_MODE=mock (default when no key is set) → returns MockLLMClient.
    - If OPENAI_API_KEY is set → returns OpenAIClient.
    - If ANTHROPIC_API_KEY is set → returns AnthropicClient.

    Tests that *require* a real API should use:
        @pytest.mark.llm
        def test_something(llm_client):
            if llm_client.provider == "mock":
                pytest.skip("Real LLM required")
    """
    from llm_eval.client import get_client

    return get_client()


@pytest.fixture(scope="session")
def real_llm_client():
    """Skip this fixture (and any test using it) when no API key is available."""
    from llm_eval.client import MockLLMClient, get_client

    client = get_client()
    if isinstance(client, MockLLMClient):
        pytest.skip("No real LLM API key configured — set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    return client


# ---------------------------------------------------------------------------
# Fixtures: RAG pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rag_pipeline(llm_client):
    """A RAGPipeline wired to the session-scoped LLM client."""
    from llm_eval.rag_pipeline import RAGPipeline, SimpleRetriever

    retriever = SimpleRetriever()
    pipeline = RAGPipeline(retriever=retriever)
    pipeline.llm = llm_client
    return pipeline


@pytest.fixture(scope="session")
def rag_results(rag_pipeline, small_qa_data):
    """
    Run the RAG pipeline on the small dataset once per session.

    This is expensive (LLM calls) so we cache it at session scope.
    """
    return rag_pipeline.run_batch(small_qa_data)


# ---------------------------------------------------------------------------
# Fixtures: RAGAS dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ragas_dataset(small_qa_data):
    """
    Build a RAGAS EvaluationDataset from the sample data.

    RAGAS 0.2+ uses datasets.Dataset with columns:
        question, answer, contexts, ground_truth
    """
    try:
        from datasets import Dataset
    except ImportError:
        pytest.skip("datasets package not installed")

    data = {
        "question": [item["question"] for item in small_qa_data],
        "answer": [item["answer"] for item in small_qa_data],
        "contexts": [item["contexts"] for item in small_qa_data],
        "ground_truth": [item["ground_truth"] for item in small_qa_data],
    }
    return Dataset.from_dict(data)


@pytest.fixture(scope="session")
def ragas_llm():
    """
    Return a LangChain-compatible LLM for RAGAS (required by RAGAS 0.2+).

    Skips if no API key is available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — RAGAS requires a real LLM")

    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    except ImportError:
        pytest.skip("langchain-openai not installed")

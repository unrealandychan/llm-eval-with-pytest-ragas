"""
Simple RAG (Retrieval-Augmented Generation) pipeline.

This is intentionally minimal — just enough to demonstrate how you would
plug RAGAS evaluation into a real pipeline.

Architecture:
    query → retrieve(context) → augmented_prompt → LLM → answer
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_eval.client import Message, get_client

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A retrieved document / context chunk."""

    content: str
    source: str = "unknown"
    score: float = 1.0


@dataclass
class RAGResult:
    """Everything produced by one RAG call — needed for RAGAS evaluation."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


# ---------------------------------------------------------------------------
# Toy in-memory retriever
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: list[Document] = [
    Document(
        content=(
            "Python's GIL (Global Interpreter Lock) is a mutex that protects access to Python "
            "objects, preventing multiple threads from executing Python bytecodes simultaneously. "
            "The GIL makes it easier to integrate with non-thread-safe C libraries and simplifies "
            "CPython's memory management, but it limits CPU-bound multithreaded performance."
        ),
        source="python-docs/concurrency",
    ),
    Document(
        content=(
            "pytest fixtures are functions that provide a fixed baseline for tests. "
            "They are declared with the @pytest.fixture decorator and injected automatically "
            "into test functions via argument names. Fixtures support scopes: function, class, "
            "module, package, and session, controlling how often they are created and torn down."
        ),
        source="pytest-docs/fixtures",
    ),
    Document(
        content=(
            "Python type hints (PEP 484) allow you to annotate variables and function signatures "
            "with expected types. They are not enforced at runtime by default, but tools like "
            "mypy, pyright, and ruff can catch type errors statically. Type hints improve "
            "readability and IDE auto-complete without affecting performance."
        ),
        source="python-docs/typing",
    ),
    Document(
        content=(
            "Generators in Python are functions that use 'yield' to return values lazily. "
            "They implement the iterator protocol without building the entire sequence in memory. "
            "Generator expressions (x for x in iterable) are a concise alternative. "
            "Use generators for large datasets or infinite sequences."
        ),
        source="python-docs/generators",
    ),
    Document(
        content=(
            "RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework "
            "for evaluating RAG pipelines. Core metrics: Faithfulness measures if the answer "
            "is grounded in the retrieved context. Answer Relevancy measures how relevant the "
            "answer is to the question. Context Recall measures how much of the ground truth "
            "is covered by the retrieved context."
        ),
        source="ragas-docs/overview",
    ),
]


class SimpleRetriever:
    """Keyword-based retriever (no embeddings needed for the demo)."""

    def __init__(self, documents: list[Document] | None = None):
        self.documents = documents or KNOWLEDGE_BASE

    def retrieve(self, query: str, top_k: int = 2) -> list[Document]:
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful technical assistant. Answer the user's question using ONLY
the provided context. If the context does not contain enough information, say
"I don't have enough information to answer this."
"""


class RAGPipeline:
    """
    End-to-end RAG pipeline wiring together retrieval and generation.

    Example:
        pipeline = RAGPipeline()
        result = pipeline.run("What is the GIL?")
        print(result.answer)
    """

    def __init__(self, retriever: SimpleRetriever | None = None, provider: str | None = None):
        self.retriever = retriever or SimpleRetriever()
        self.llm = get_client(provider=provider)  # type: ignore[arg-type]

    def run(self, question: str, ground_truth: str = "", top_k: int = 2) -> RAGResult:
        docs = self.retriever.retrieve(question, top_k=top_k)
        contexts = [doc.content for doc in docs]

        context_block = "\n\n---\n\n".join(
            f"[Source: {doc.source}]\n{doc.content}" for doc in docs
        )
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]
        response = self.llm.chat(messages)

        return RAGResult(
            question=question,
            answer=response.content,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    def run_batch(self, qa_pairs: list[dict]) -> list[RAGResult]:
        """Run the pipeline on multiple Q&A pairs."""
        results = []
        for pair in qa_pairs:
            result = self.run(
                question=pair["question"],
                ground_truth=pair.get("ground_truth", ""),
            )
            results.append(result)
        return results

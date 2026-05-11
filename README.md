# llm-eval-with-pytest-ragas

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-orange.svg)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/unrealandychan/llm-eval-with-pytest-ragas/actions/workflows/ci.yml/badge.svg)](https://github.com/unrealandychan/llm-eval-with-pytest-ragas/actions)
[![PyCon HK 2026](https://img.shields.io/badge/PyCon%20HK-2026-red.svg)](https://pycon.hk)

> **"How do you know if your AI answered correctly?"**  
> A practical, beginner-friendly guide to evaluating LLM applications using **pytest** and **RAGAS**.

Demo repository for the PyCon HK 2026 lightning talk:  
🎤 *"Evaluating LLMs: 點知 AI 答得啱唔啱？"*

---

## 🤔 The Problem

You've built a RAG (Retrieval-Augmented Generation) chatbot. It seems to work.
But how do you *know* it's good? How do you catch regressions before your users do?

```python
# Traditional unit test — trivial to assert:
assert add(2, 3) == 5

# LLM test — what do you assert against?
response = llm.answer("What is the Python GIL?")
assert response == ???  # 🤷 LLMs are non-deterministic!
```

The answer: **don't compare strings — measure quality**.

When you do **not** have an explicit reference answer, use an
**agentic no-reference workflow** that evaluates:
- whether the answer covers the question intent,
- whether it stays grounded in retrieved context,
- whether uncertainty/refusal behavior is appropriate for the context quality.

---

## 🧪 What is RAGAS?

[RAGAS](https://docs.ragas.io) (Retrieval Augmented Generation Assessment) is an
open-source framework that evaluates RAG pipelines using an **LLM-as-judge** approach.

Instead of needing a massive labeled test set, RAGAS uses your LLM to score
its own outputs along three core dimensions:

| Metric | Question it answers | Analogy |
|--------|-------------------|---------|
| **Faithfulness** | Did the model only say things supported by the retrieved context? | Open-book exam — answers must come from the notes |
| **Answer Relevancy** | Did the answer actually address the question? | Did you answer what was asked, or go off-topic? |
| **Context Recall** | Did the retriever fetch enough information to answer correctly? | Did the library have the right books? |

All scores are **0.0 – 1.0**, higher is better.

---

## ⚡ Quick Start

### 1. Clone and install

```bash
git clone https://github.com/unrealandychan/llm-eval-with-pytest-ragas.git
cd llm-eval-with-pytest-ragas

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run tests immediately (no API key needed!)

```bash
# Mock mode — instant, free, no API calls
LLM_EVAL_MODE=mock pytest -m "not llm and not ragas" -v
```

Expected output:
```
tests/test_custom_metrics.py::TestResponseLength::test_normal_answer_passes PASSED
tests/test_custom_metrics.py::TestNoExplicitRefusal::test_good_answer_passes PASSED
tests/test_custom_metrics.py::TestDatasetQuality::test_minimum_dataset_size PASSED
tests/test_integration.py::test_rag_pipeline_returns_result PASSED
...
✅ 20 passed in 0.42s
```

### 3. Run RAGAS tests (needs OpenAI API key)

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

pytest -m ragas -v
```

---

## 📊 Metrics Explained

### Faithfulness — "Did the model make things up?"

```
Context: "The GIL is a mutex that protects Python objects."

Good answer: "The GIL is a mutex preventing parallel thread execution."
             ↑ Everything is supported by context ✅ → Faithfulness: 1.0

Bad answer:  "The GIL is a mutex. It was introduced in Python 2.0."
             ↑ Supported     ↑ NOT in context → hallucination! ❌ → Faithfulness: 0.5
```

**Threshold:** ≥ 0.7 for production systems.

### Answer Relevancy — "Did the model stay on topic?"

RAGAS generates synthetic questions from the answer and checks if they match the original question.
Off-topic answers produce synthetic questions that don't match → low score.

**Threshold:** ≥ 0.7

### Context Recall — "Did we retrieve the right documents?"

```
Ground truth: "GIL is a mutex. It prevents parallel threads. It exists for memory safety."
                     ↑                     ↑                          ↑
Retrieved context covers: sentence 2 only

Context Recall = 1/3 = 0.33 ❌ → Fix your retriever!
```

**Threshold:** ≥ 0.7

### Agentic No-Reference Workflow — "How do we evaluate with no explicit answer?"

Sometimes you only have:
- the user question,
- retrieved context,
- model output,

but no trusted `ground_truth`.

In that case, this repo now includes `run_agentic_workflow(...)`, a multi-step
quality gate built from deterministic checks:

1. **Response sanity** (length)
2. **Question coverage** (key terms from question appear in answer)
3. **Context grounding** (answer tokens overlap with retrieved context)
4. **Appropriate uncertainty** (refusal behavior matches context sufficiency)

This gives a robust baseline for CI even before you curate explicit labels.

---

## 🗂 Project Structure

```
llm-eval-with-pytest-ragas/
├── src/llm_eval/
│   ├── client.py          # Unified LLM client (OpenAI, Anthropic, Mock)
│   ├── rag_pipeline.py    # Simple RAG pipeline to evaluate
│   └── metrics.py         # Heuristic + agentic no-reference metrics
│
├── tests/
│   ├── conftest.py              # pytest fixtures (dataset, LLM client, RAGAS)
│   ├── test_faithfulness.py     # Faithfulness tests
│   ├── test_answer_relevancy.py # Answer relevancy tests
│   ├── test_context_recall.py   # Context recall tests
│   ├── test_custom_metrics.py   # Heuristic tests (no API key)
│   └── test_integration.py      # End-to-end pipeline tests
│
├── examples/
│   ├── basic_eval.py       # Standalone eval (30 lines)
│   ├── rag_eval.py         # Full RAG pipeline eval with Rich output
│   └── custom_metrics.py   # How to write your own metrics
│
├── datasets/
│   └── sample_qa.json      # 10 Python/programming Q&A pairs with context
│
└── docs/
    ├── getting-started.md
    ├── metrics-explained.md
    └── pycon-hk-2026-slides-outline.md   # 🎤 Talk outline
```

---

## 🏃 Running Tests

### pytest marks

This project uses custom marks to categorize tests:

| Mark | Meaning | Needs API key? |
|------|---------|----------------|
| *(no mark)* | Pure Python, deterministic | ❌ No |
| `@pytest.mark.integration` | Uses the RAG pipeline | ❌ No (mock mode) |
| `@pytest.mark.llm` | Calls a real LLM API | ✅ Yes |
| `@pytest.mark.ragas` | Uses RAGAS framework | ✅ Yes |

### Common commands

```bash
# Fast: no API key, runs in < 1 second
pytest -m "not llm and not ragas" -v

# Integration tests (mock mode)
LLM_EVAL_MODE=mock pytest tests/test_integration.py -v

# Only RAGAS tests
pytest -m ragas -v

# Everything (needs API key)
OPENAI_API_KEY=sk-... pytest -v

# Run a specific test file
pytest tests/test_custom_metrics.py -v

# Show slowest tests
pytest --durations=5
```

---

## 🔧 Running Without an API Key

Set `LLM_EVAL_MODE=mock` in your `.env` or environment.

The `MockLLMClient` returns deterministic canned responses — perfect for:
- Local development
- CI/CD without secrets
- Learning the framework

```python
# This is what mock mode does internally:
class MockLLMClient:
    def chat(self, messages):
        return LLMResponse(content="[MOCK] Canned response", model="mock-gpt")
```

Tests decorated with `@pytest.mark.llm` or `@pytest.mark.ragas` will
automatically skip if no real API key is detected.

---

## 🤖 Writing Your Own Tests

### Basic pattern

```python
# tests/test_my_rag.py
import pytest
from llm_eval.metrics import context_grounding

def test_grounding(small_qa_data):
    """All answers must be grounded in their contexts."""
    for item in small_qa_data:
        result = context_grounding(
            answer=item["answer"],
            contexts=item["contexts"],
            threshold=0.3,
        )
        assert result.passed, f"Low grounding for: {item['question']}\n{result}"
```

### RAGAS pattern

```python
@pytest.mark.ragas
@pytest.mark.llm
def test_faithfulness(ragas_dataset, ragas_llm):
    from ragas import evaluate
    from ragas.metrics import faithfulness

    result = evaluate(ragas_dataset, metrics=[faithfulness], llm=ragas_llm)
    assert result["faithfulness"] >= 0.7, f"Score: {result['faithfulness']:.2f}"
```

### Custom metric

```python
from llm_eval.metrics import MetricResult

def my_metric(answer: str) -> MetricResult:
    has_example = "```" in answer
    return MetricResult(
        name="has_code_example",
        score=1.0 if has_example else 0.0,
        passed=has_example,
        details="Found code block" if has_example else "Missing code example",
    )
```

### No-reference (agentic) pattern

```python
from llm_eval.metrics import run_agentic_workflow

result = run_agentic_workflow(
    question=item["question"],
    answer=item["answer"],
    contexts=item["contexts"],
)
assert result.passed, f"Agentic score too low: {result.overall_score:.2f}"
```

---

## 🔄 CI/CD Integration

### GitHub Actions example

```yaml
# .github/workflows/ci.yml
jobs:
  test-fast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      
      # No API key needed — runs in CI for free
      - run: LLM_EVAL_MODE=mock pytest -m "not llm and not ragas" -v

  test-ragas:
    # Only run on schedule (expensive — uses real API)
    if: github.event_name == 'schedule'
    steps:
      - run: pytest -m ragas -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**Cost-saving strategy:**
- ✅ Run heuristic tests on **every push** (free, fast)
- ✅ Run RAGAS on **weekly schedule** or pre-release only (costs API credits)

---

## 📦 Installation as a Library

```bash
pip install -e .          # Core only
pip install -e ".[dev]"   # Core + pytest + dev tools
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `ragas>=0.2` | Core evaluation framework |
| `openai>=1.0` | OpenAI client |
| `anthropic` | Anthropic/Claude client |
| `langchain-openai` | LangChain LLM wrapper for RAGAS |
| `python-dotenv` | Load `.env` files |
| `rich` | Pretty terminal output |
| `datasets` | HuggingFace datasets (RAGAS input format) |
| `pytest>=8.0` | Test runner |
| `pytest-asyncio` | Async test support |

---

## 📚 Sample Dataset

The project includes `datasets/sample_qa.json` with 10 Python/programming Q&A pairs,
each containing:

- `question` — the user's question
- `contexts` — retrieved document chunks (what the RAG system found)
- `answer` — the model's generated answer
- `ground_truth` — the reference/expected answer (for context recall)

Topics covered:
1. Python GIL (Global Interpreter Lock)
2. pytest fixtures and scopes
3. Python type hints
4. Generators and `yield`
5. RAGAS metrics
6. Lists vs. tuples
7. asyncio event loop
8. Decorators
9. `is` vs `==`
10. Memory management (reference counting + GC)

---

## 🎤 PyCon HK 2026

This repository is the demo resource for the lightning talk:

> **"Evaluating LLMs: 點知 AI 答得啱唔啱？"**  
> *How do you know if your AI answered correctly?*

📄 [Full talk outline with slide notes](docs/pycon-hk-2026-slides-outline.md)

The talk covers:
- Why traditional unit tests don't work for LLMs
- RAGAS metrics explained with simple analogies (in English + Cantonese)
- Live pytest demo running in under 1 second
- CI/CD strategy for LLM quality gates

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-metric`
3. Add tests for your changes
4. Run: `pytest -m "not llm and not ragas"` — must pass
5. Open a pull request

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🔗 Resources

- [RAGAS Documentation](https://docs.ragas.io)
- [pytest Documentation](https://docs.pytest.org)
- [OpenAI API](https://platform.openai.com/docs)
- [Anthropic API](https://docs.anthropic.com)
- [PyCon HK](https://pycon.hk)

---

*Built with ❤️ for PyCon HK 2026 — 香港 Python 社群*

# Getting Started

## Prerequisites

- Python 3.11+
- `pip` or [`uv`](https://github.com/astral-sh/uv) (recommended)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/unrealandychan/llm-eval-with-pytest-ragas.git
cd llm-eval-with-pytest-ragas
```

### 2. Create a virtual environment

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate

# Or using standard venv
python -m venv .venv && source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Install package + dev dependencies
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Running Tests

### Without API key (mock mode — instant, no cost)

```bash
# All non-LLM tests
pytest -m "not llm and not ragas" -v

# Just the custom metrics tests
pytest tests/test_custom_metrics.py -v

# Integration tests in mock mode
pytest tests/test_integration.py -v
```

### With OpenAI API key

```bash
export OPENAI_API_KEY=sk-...

# Run RAGAS tests
pytest -m ragas -v

# Run everything
pytest -v
```

## Project Layout

```
src/llm_eval/
├── client.py        # LLM client (OpenAI, Anthropic, Mock)
├── rag_pipeline.py  # RAG pipeline to evaluate
└── metrics.py       # Custom heuristic metrics

tests/
├── conftest.py              # Shared fixtures
├── test_faithfulness.py     # Faithfulness metric tests
├── test_answer_relevancy.py # Answer relevancy tests
├── test_context_recall.py   # Context recall tests
├── test_custom_metrics.py   # Heuristic metric tests (no API)
└── test_integration.py      # End-to-end tests

examples/
├── basic_eval.py     # Standalone evaluation
├── rag_eval.py       # Full RAG pipeline eval
└── custom_metrics.py # Custom metric patterns

datasets/
└── sample_qa.json    # 10 Python Q&A pairs with contexts
```

## pytest Marks

| Mark | Description |
|------|-------------|
| `@pytest.mark.llm` | Calls a real LLM API |
| `@pytest.mark.ragas` | Uses RAGAS evaluation |
| `@pytest.mark.integration` | End-to-end pipeline test |

Skip specific marks:
```bash
pytest -m "not llm"        # No API calls
pytest -m "not ragas"      # No RAGAS (faster)
pytest -m "integration"    # Only integration tests
```

## Evaluating Without Explicit Answers

If you do not have `ground_truth` labels yet, use the built-in agentic workflow:

```python
from llm_eval.metrics import run_agentic_workflow

result = run_agentic_workflow(
    question="What is the Python GIL?",
    answer="The GIL is a lock that prevents multiple Python threads from executing bytecode at once.",
    contexts=["Python's GIL is a mutex preventing simultaneous bytecode execution by threads."],
)

print(result.overall_score, result.passed)
```

This gives you deterministic quality gates for CI while you gradually build a
labeled evaluation dataset.

"""
custom_metrics.py — demonstrate how to write your own evaluation metrics.

This example shows three patterns:
1. Rule-based metric (no LLM needed)
2. LLM-as-judge metric (uses an LLM to evaluate)
3. Combining metrics into a composite score

Run:
    python examples/custom_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from llm_eval.client import get_client, Message
from llm_eval.metrics import MetricResult

console = Console()


# ---------------------------------------------------------------------------
# Pattern 1: Rule-based metric — fast, deterministic, no API calls
# ---------------------------------------------------------------------------


def metric_contains_code_example(answer: str) -> MetricResult:
    """
    Custom metric: does the answer include a code example?

    For a coding assistant, answers without code examples may be incomplete.
    """
    has_code = "```" in answer or "`" in answer
    return MetricResult(
        name="contains_code_example",
        score=1.0 if has_code else 0.0,
        passed=has_code,
        details="Found inline/block code" if has_code else "No code example found",
    )


def metric_no_pii(answer: str) -> MetricResult:
    """
    Custom metric: does the answer accidentally reveal PII?

    Simplified check — real implementation would use regex/NER.
    """
    import re

    # Very simplified PII patterns for demo purposes
    patterns = [
        (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "Possible full name"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN pattern"),
        (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "Email address"),
    ]

    for pattern, label in patterns:
        if re.search(pattern, answer):
            return MetricResult(
                name="no_pii",
                score=0.0,
                passed=False,
                details=f"Possible PII detected: {label}",
            )
    return MetricResult(name="no_pii", score=1.0, passed=True, details="No PII patterns found")


# ---------------------------------------------------------------------------
# Pattern 2: LLM-as-judge — uses an LLM to score the answer
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an expert evaluator. Rate the following answer on a scale of 0 to 10
for technical accuracy and clarity.

Question: {question}
Answer: {answer}

Respond with ONLY a number between 0 and 10. Nothing else.
"""


def metric_llm_judge(question: str, answer: str, client=None) -> MetricResult:
    """
    LLM-as-judge metric: ask an LLM to rate the answer quality.

    This is powerful but costs API credits. Use sparingly in CI.
    """
    if client is None:
        client = get_client()

    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    response = client.complete(prompt)

    # Parse score from response
    try:
        # Extract first number found in response
        import re
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", response)
        raw_score = float(numbers[0]) if numbers else 5.0
        score = min(max(raw_score / 10.0, 0.0), 1.0)  # Normalize to 0-1
    except (ValueError, IndexError):
        score = 0.5  # Default if parsing fails

    passed = score >= 0.7
    return MetricResult(
        name="llm_judge",
        score=score,
        passed=passed,
        details=f"LLM rated: {score * 10:.1f}/10 (provider: {client.provider})",
    )


# ---------------------------------------------------------------------------
# Pattern 3: Composite metric
# ---------------------------------------------------------------------------


@dataclass
class CompositeEvalResult:
    individual: list[MetricResult]
    composite_score: float
    passed: bool

    def __str__(self) -> str:
        lines = [f"Composite Score: {self.composite_score:.3f} ({'PASS' if self.passed else 'FAIL'})"]
        for m in self.individual:
            lines.append(f"  {m}")
        return "\n".join(lines)


def composite_eval(
    question: str,
    answer: str,
    contexts: list[str],
    threshold: float = 0.6,
) -> CompositeEvalResult:
    """Run multiple metrics and combine into a single pass/fail decision."""
    from llm_eval.metrics import context_grounding, response_length_ok, no_explicit_refusal

    metrics = [
        response_length_ok(answer),
        no_explicit_refusal(answer),
        context_grounding(answer, contexts),
        metric_contains_code_example(answer),
        metric_no_pii(answer),
    ]

    score = sum(m.score for m in metrics) / len(metrics)
    return CompositeEvalResult(
        individual=metrics,
        composite_score=score,
        passed=score >= threshold,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main():
    console.print("\n[bold cyan]Custom Metrics Demo[/bold cyan]\n")

    question = "How do I use a Python generator?"
    answer = """
Generators in Python use the `yield` keyword to produce values lazily.

```python
def count_up(n):
    for i in range(n):
        yield i

for num in count_up(5):
    print(num)  # 0, 1, 2, 3, 4
```

This avoids loading the entire sequence into memory.
"""
    contexts = [
        "Generators in Python are functions that use 'yield' to return values lazily. "
        "They implement the iterator protocol without building the entire sequence in memory."
    ]

    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[bold]Answer:[/bold] {answer.strip()[:100]}...\n")

    result = composite_eval(question=question, answer=answer, contexts=contexts)
    console.print(result)

    # LLM-as-judge demo
    console.print("\n[bold]LLM-as-Judge:[/bold]")
    client = get_client()
    judge_result = metric_llm_judge(question, answer, client)
    console.print(f"  {judge_result}")


if __name__ == "__main__":
    main()

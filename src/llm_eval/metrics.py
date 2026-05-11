"""
Custom metric helpers for LLM evaluation.

These metrics can be used standalone (no RAGAS needed) and are useful
for quick sanity checks and CI gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MetricResult:
    name: str
    score: float          # 0.0 – 1.0
    passed: bool
    details: str = ""

    def __repr__(self) -> str:  # noqa: D105
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} [{self.name}] score={self.score:.3f} | {self.details}"


# ---------------------------------------------------------------------------
# Length metrics
# ---------------------------------------------------------------------------


def response_length_ok(
    answer: str,
    min_words: int = 10,
    max_words: int = 500,
) -> MetricResult:
    """Check that the answer is within a reasonable word-count range."""
    words = len(answer.split())
    passed = min_words <= words <= max_words
    return MetricResult(
        name="response_length",
        score=1.0 if passed else 0.0,
        passed=passed,
        details=f"word_count={words} (expected {min_words}–{max_words})",
    )


# ---------------------------------------------------------------------------
# Hallucination heuristics
# ---------------------------------------------------------------------------

HALLUCINATION_PHRASES = [
    r"\bi don't know\b",
    r"\bi do not know\b",
    r"\bi'm not sure\b",
    r"\bi am not sure\b",
    r"\bi cannot answer\b",
    r"\bi can't answer\b",
    r"\bno information\b",
    r"\bnot enough information\b",
]


def no_explicit_refusal(answer: str) -> MetricResult:
    """
    Check the model didn't refuse to answer.

    Not a hallucination detector, but catches obvious failure modes where
    the model says "I don't know" when context was provided.
    """
    answer_lower = answer.lower()
    for pattern in HALLUCINATION_PHRASES:
        if re.search(pattern, answer_lower):
            return MetricResult(
                name="no_explicit_refusal",
                score=0.0,
                passed=False,
                details=f"Answer contains refusal pattern: '{pattern}'",
            )
    return MetricResult(name="no_explicit_refusal", score=1.0, passed=True)


# ---------------------------------------------------------------------------
# Keyword coverage
# ---------------------------------------------------------------------------


def keyword_coverage(answer: str, keywords: list[str], threshold: float = 0.5) -> MetricResult:
    """
    Check that the answer mentions at least `threshold` fraction of expected keywords.

    Useful for domain-specific tests where certain terms *must* appear.
    """
    answer_lower = answer.lower()
    hits = [kw for kw in keywords if kw.lower() in answer_lower]
    score = len(hits) / len(keywords) if keywords else 1.0
    passed = score >= threshold
    return MetricResult(
        name="keyword_coverage",
        score=score,
        passed=passed,
        details=f"covered {len(hits)}/{len(keywords)} keywords: {hits}",
    )


# ---------------------------------------------------------------------------
# Context grounding (simple overlap)
# ---------------------------------------------------------------------------


def context_grounding(answer: str, contexts: list[str], threshold: float = 0.3) -> MetricResult:
    """
    Rough measure: what fraction of answer tokens appear in the retrieved context?

    This is a cheap alternative to RAGAS faithfulness — handy for CI without API calls.
    """
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    context_text = " ".join(contexts).lower()
    context_tokens = set(re.findall(r"\b\w+\b", context_text))

    # Remove stop words
    stopwords = {"the", "a", "an", "is", "in", "it", "of", "to", "and", "or", "for", "with"}
    answer_tokens -= stopwords
    context_tokens -= stopwords

    if not answer_tokens:
        return MetricResult(name="context_grounding", score=0.0, passed=False, details="empty answer")

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    passed = overlap >= threshold
    return MetricResult(
        name="context_grounding",
        score=overlap,
        passed=passed,
        details=f"overlap={overlap:.2%} (threshold={threshold:.0%})",
    )


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def run_all_metrics(
    answer: str,
    contexts: list[str],
    keywords: list[str] | None = None,
) -> list[MetricResult]:
    """Run all built-in metrics and return the results."""
    results = [
        response_length_ok(answer),
        no_explicit_refusal(answer),
        context_grounding(answer, contexts),
    ]
    if keywords:
        results.append(keyword_coverage(answer, keywords))
    return results


def overall_score(results: list[MetricResult]) -> float:
    """Average score across all metrics."""
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)

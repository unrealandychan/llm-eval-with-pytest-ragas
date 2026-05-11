# Metrics Explained

## What is RAGAS?

RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework
for evaluating RAG (Retrieval-Augmented Generation) pipelines.

Traditional ML metrics (accuracy, F1) don't work well for free-text LLM outputs.
RAGAS provides reference-free metrics that use an LLM-as-judge to score quality.

---

## Core RAGAS Metrics

### 1. Faithfulness

**What it measures:** Did the model only say things supported by the retrieved context?

**Simple analogy:** 🎓 *An open-book exam student should only write answers from their notes.
If they invent facts not in the notes — that's a faithfulness failure.*

**Score range:** 0.0 – 1.0 (higher is better)

**How RAGAS calculates it:**
1. Extract atomic claims from the answer
2. For each claim, ask the LLM: "Is this claim supported by the context?"
3. Score = supported claims / total claims

**Example:**
```
Context: "The GIL prevents multiple threads from running simultaneously."
Answer: "The GIL prevents parallel threads AND was introduced in Python 2.0."
         ↑ supported                            ↑ NOT in context → hallucination!
Faithfulness = 1/2 = 0.5
```

**Threshold recommendation:** ≥ 0.7

---

### 2. Answer Relevancy

**What it measures:** Does the answer actually address the question?

**Simple analogy:** 🎯 *Asking "What time does the library close?" and getting a 
history of the Roman Empire — technically words, but zero relevancy.*

**Score range:** 0.0 – 1.0 (higher is better)

**How RAGAS calculates it:**
1. Generate N synthetic questions from the answer
2. Compute cosine similarity between synthetic questions and original question
3. Score = average similarity

**Example:**
```
Question: "What is a Python generator?"
Answer: "Generators use yield and are memory-efficient for large sequences."
→ Synthetic Q: "What keyword do generators use?" ✅ Similar
→ Synthetic Q: "What are generators useful for?" ✅ Similar
Answer Relevancy = high ✅

Answer: "Python was created by Guido van Rossum in 1991."
→ Completely off-topic → low relevancy ❌
```

**Threshold recommendation:** ≥ 0.7

---

### 3. Context Recall

**What it measures:** Did the retriever fetch documents covering the ground truth?

**Simple analogy:** 📚 *If the correct answer requires knowing facts A, B, and C,
but you only retrieved a document with fact A — you'll miss the rest.*

**Score range:** 0.0 – 1.0 (higher is better)

**How RAGAS calculates it:**
1. Break ground truth into atomic sentences
2. For each sentence, ask: "Is this covered by the retrieved context?"
3. Score = covered sentences / total sentences

**Example:**
```
Ground truth: "GIL is a mutex. It prevents parallel threads. It exists for memory safety."
Context retrieved: "The GIL prevents parallel threads."
                    ↑ covers sentence 2 only
Context Recall = 1/3 ≈ 0.33 ❌ (retriever is missing documents!)
```

**Threshold recommendation:** ≥ 0.7

---

## Custom Heuristic Metrics (No LLM needed)

These run instantly and cost nothing — great for CI pipelines.

### Response Length

```python
from llm_eval.metrics import response_length_ok
result = response_length_ok(answer, min_words=10, max_words=500)
```

### No Explicit Refusal

Catches answers like "I don't know" or "I cannot answer this."

```python
from llm_eval.metrics import no_explicit_refusal
result = no_explicit_refusal(answer)
```

### Keyword Coverage

Check that domain-specific terms appear in the answer.

```python
from llm_eval.metrics import keyword_coverage
result = keyword_coverage(answer, keywords=["GIL", "mutex", "thread"], threshold=0.6)
```

### Context Grounding (Heuristic)

Token-overlap approximation of faithfulness — no API calls.

```python
from llm_eval.metrics import context_grounding
result = context_grounding(answer, contexts, threshold=0.3)
```

---

## Choosing Thresholds

| Use Case | Faithfulness | Answer Relevancy | Context Recall |
|----------|-------------|-----------------|----------------|
| Customer-facing chatbot | ≥ 0.85 | ≥ 0.80 | ≥ 0.75 |
| Internal knowledge base | ≥ 0.75 | ≥ 0.70 | ≥ 0.70 |
| Research / exploration | ≥ 0.60 | ≥ 0.60 | ≥ 0.60 |
| Demo / prototype | ≥ 0.50 | ≥ 0.50 | ≥ 0.50 |

---

## When Metrics are Low: What to Fix

| Low Metric | Likely Problem | Fix |
|-----------|---------------|-----|
| Faithfulness | Model hallucinating | Better system prompt; smaller/cleaner context |
| Answer Relevancy | Model going off-topic | Stricter system prompt; fine-tuning |
| Context Recall | Wrong docs retrieved | Better embeddings; re-ranker; hybrid search |

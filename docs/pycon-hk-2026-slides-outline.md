# PyCon HK 2026 — Lightning Talk Outline

## Title: "Evaluating LLMs: 點知 AI 答得啱唔啱？"
## (How do you know if your AI answered correctly?)

**Format:** Lightning Talk — 5 minutes  
**Speaker:** Andy Chan  
**Event:** PyCon HK 2026  
**Repo:** https://github.com/unrealandychan/llm-eval-with-pytest-ragas

---

## Talk Summary

We all build LLM-powered apps, but how do we *know* they're working?
Unit tests for LLMs are different — you can't just `assert answer == expected`.
This talk shows how to use **pytest + RAGAS** to set quality gates for RAG systems,
so you ship with confidence instead of hope.

---

## Slide-by-Slide Breakdown

---

### Slide 1 — Title (20 seconds)

**Visual:** 大字標題 (Big title text)

```
"Evaluating LLMs: 點知 AI 答得啱唔啱？"
```

**Talking points:**
- 大家好！I'm Andy.
- 你哋係咪 build 緊 AI chatbot 或者 RAG app？
- (Are you building AI chatbots or RAG apps?)
- 今日我想問一個問題：你點知 AI 答得啱唔啱？
- (Today I want to ask: how do you know if your AI answered correctly?)

---

### Slide 2 — The Problem (45 seconds)

**Visual:** Code snippet + question mark

```python
# Traditional unit test — easy!
def add(a, b): return a + b
assert add(2, 3) == 5  ✅

# LLM unit test — how??
response = llm.answer("What is the GIL?")
assert response == ???  🤔
```

**Talking points:**
- Traditional tests are deterministic — same input, same output.
- LLM responses are probabilistic — every run can be different!
- 我哋唔可以 `assert response == expected_answer`
- (We can't just `assert response == expected_answer`)
- So... what do we assert?

**Key insight to emphasize:**
> "We need metrics, not exact matches."

---

### Slide 3 — Introducing RAGAS (60 seconds)

**Visual:** Three metric cards

```
RAGAS = Retrieval Augmented Generation Assessment

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Faithfulness   │  │Answer Relevancy │  │ Context Recall  │
│                 │  │                 │  │                 │
│ 係咪淨係講      │  │ 有冇答到條問題  │  │ 搵返嚟嘅資料   │
│ context度嘅嘢？ │  │  (on-topic?)    │  │ 夠唔夠全面？   │
│ (grounded?)     │  │                 │  │ (complete?)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Talking points:**
- **Faithfulness** — 好似開卷考試：答案要從資料度嚟，唔可以亂噏
  - (Like an open-book exam: answer must come from the materials, no making things up)
- **Answer Relevancy** — 係咪答到個問題？定係答其他嘢？
  - (Did it answer the question, or go off-topic?)
- **Context Recall** — 我哋 retrieve 返嚟嘅資料，夠唔夠答到個問題？
  - (Did we retrieve enough information to answer correctly?)

---

### Slide 4 — Live Code Demo (90 seconds)

**Visual:** Terminal + code split screen

**Step 1: Show the test**
```python
# tests/test_faithfulness.py
import pytest
from ragas.metrics import faithfulness
from ragas import evaluate

@pytest.mark.ragas
def test_faithfulness_above_threshold(ragas_dataset, ragas_llm):
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
    )
    score = result["faithfulness"]
    assert score >= 0.7, f"Faithfulness: {score:.2f} < 0.7 ❌"
```

**Talking points:**
- 睇！就係普通 pytest test！
- (Look — it's just a regular pytest test!)
- `ragas_dataset` 係一個 fixture — questions, contexts, answers
- 如果 score 低過 0.7，build 就會 fail！
- (If score < 0.7, the build fails!)

**Step 2: Run it live**
```bash
$ pytest tests/test_faithfulness.py -v -m "not ragas"
# (Run without API key first to show mock mode)

tests/test_faithfulness.py::test_context_grounding_heuristic PASSED ✅
tests/test_faithfulness.py::test_answer_not_empty PASSED ✅
tests/test_faithfulness.py::test_contexts_not_empty PASSED ✅
```

**Step 3: Show fast heuristic metric**
```python
# No API key needed!
from llm_eval.metrics import context_grounding

result = context_grounding(
    answer="The GIL is a mutex that prevents parallel threads.",
    contexts=["Python's GIL prevents multiple threads..."],
    threshold=0.3,
)
print(result)
# ✅ PASS [context_grounding] score=0.714 | overlap=71%
```

---

### Slide 5 — CI/CD Integration (30 seconds)

**Visual:** GitHub Actions YAML (short snippet)

```yaml
# .github/workflows/eval.yml
- name: Run LLM eval tests
  run: |
    pytest -m "not llm and not ragas" -v  # Fast, free
    
# With API key (scheduled weekly):
- name: Run RAGAS evaluation
  run: pytest -m ragas -v
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**Talking points:**
- 唔需要每次 CI 都跑 RAGAS（太貴！）
- (Don't run RAGAS on every CI push — too expensive!)
- 策略：快速 heuristic tests 每次跑，RAGAS 每週跑一次
- (Strategy: fast heuristic tests every push, RAGAS weekly)

---

### Slide 6 — Key Takeaways + QR Code (35 seconds)

**Visual:** QR code linking to repo + 3 bullet points

```
🔑 Key Takeaways:

1. LLM eval needs METRICS not exact matches
   (用分數，唔係 == 比較)

2. pytest + RAGAS = quality gates for RAG
   (質量關卡：score < threshold → build fail)

3. Start with free heuristics, add RAGAS when ready
   (先用免費 heuristic，後加 RAGAS)

📦 github.com/unrealandychan/llm-eval-with-pytest-ragas
```

**Talking points:**
- Scan the QR code — clone the repo, run the tests NOW!
- 唔需要 API key 先可以試！Mock mode 即係得！
- (No API key needed to start — mock mode works!)
- 有問題？GitHub issues 或者之後搵我！

---

## Timing Breakdown

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title + hook | 0:20 |
| 2 | The problem | 0:45 |
| 3 | RAGAS metrics | 1:00 |
| 4 | Live code demo | 1:30 |
| 5 | CI/CD | 0:30 |
| 6 | Takeaways + QR | 0:35 |
| **Total** | | **5:00** |

---

## Props / Setup Needed

- [ ] Terminal open with repo cloned
- [ ] `pytest tests/test_custom_metrics.py -v` ready to run
- [ ] `.env` set to mock mode (so it runs without Wi-Fi)
- [ ] Browser tab open: GitHub repo
- [ ] QR code slide prepared

---

## Backup Slides (if time permits)

### RAGAS Metric Deep Dive — Faithfulness

```
Context: "The GIL is a mutex protecting Python objects."

Answer: "The GIL is a mutex. It was created in 1992."
         ↑ ✅ Supported          ↑ ❌ NOT in context!

Faithfulness = 1 supported / 2 claims = 0.50 ❌
```

### Why Not Just Use Human Eval?

| | Human Eval | RAGAS |
|--|-----------|-------|
| Cost | 💸💸💸 | 💸 |
| Speed | Days | Minutes |
| Consistency | Variable | Consistent |
| Scale | Hard | Easy |
| CI/CD | ❌ | ✅ |

---

## Glossary (for audience members unfamiliar with Cantonese)

| Cantonese | English |
|-----------|---------|
| 點知 | How do you know |
| 答得啱唔啱 | Answered correctly or not |
| 唔可以 | Cannot / should not |
| 係咪 | Is it? / Are you? |
| 夠唔夠 | Enough or not |
| 唔需要 | Don't need to |
| 即係得 | It works |

---

*準備好！Let's show Hong Kong how to test AI properly! 🚀*

"""
test_memory.py — Tests for the two-tier SQLite memory layer.

All tests use the default in-memory SQLite database (:memory:), so they
run instantly with no file-system side effects and no API key required.

Run:
    pytest tests/test_memory.py -v
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Short-term memory
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    def test_add_and_retrieve(self):
        from llm_eval.memory import MemoryEntry, ShortTermMemory

        stm = ShortTermMemory(capacity=5)
        entry = MemoryEntry(session_id="s1", entry_type="turn", role="user", content="Hello")
        stm.add(entry)
        assert len(stm) == 1
        result = stm.get_recent()
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_capacity_eviction(self):
        from llm_eval.memory import MemoryEntry, ShortTermMemory

        stm = ShortTermMemory(capacity=3)
        for i in range(5):
            stm.add(MemoryEntry(session_id="s1", entry_type="turn", role="user", content=str(i)))
        # Only the last 3 should remain
        assert len(stm) == 3
        recent = stm.get_recent()
        assert [e.content for e in recent] == ["2", "3", "4"]

    def test_get_recent_n(self):
        from llm_eval.memory import MemoryEntry, ShortTermMemory

        stm = ShortTermMemory(capacity=10)
        for i in range(6):
            stm.add(MemoryEntry(session_id="s1", entry_type="turn", role="user", content=str(i)))
        last_two = stm.get_recent(2)
        assert len(last_two) == 2
        assert last_two[-1].content == "5"

    def test_clear(self):
        from llm_eval.memory import MemoryEntry, ShortTermMemory

        stm = ShortTermMemory(capacity=5)
        stm.add(MemoryEntry(session_id="s1", entry_type="turn", role="user", content="hi"))
        stm.clear()
        assert len(stm) == 0


# ---------------------------------------------------------------------------
# Long-term memory (SQLite :memory:)
# ---------------------------------------------------------------------------


class TestLongTermMemory:
    def test_save_and_count(self):
        from llm_eval.memory import LongTermMemory, MemoryEntry

        ltm = LongTermMemory()
        entry = MemoryEntry(session_id="s1", entry_type="turn", role="user", content="test")
        saved = ltm.save(entry)
        assert saved.id is not None
        assert ltm.count() == 1
        ltm.close()

    def test_get_session_turns(self):
        from llm_eval.memory import LongTermMemory, MemoryEntry

        ltm = LongTermMemory()
        for role, content in [("user", "Q1"), ("assistant", "A1"), ("user", "Q2")]:
            ltm.save(MemoryEntry(session_id="s1", entry_type="turn", role=role, content=content))
        # Unrelated session
        ltm.save(MemoryEntry(session_id="s2", entry_type="turn", role="user", content="other"))

        turns = ltm.get_session_turns("s1")
        assert len(turns) == 3
        assert turns[0].content == "Q1"
        ltm.close()

    def test_search_by_content(self):
        from llm_eval.memory import LongTermMemory, MemoryEntry

        ltm = LongTermMemory()
        ltm.save(MemoryEntry(session_id="s1", entry_type="fact", content="Python GIL details"))
        ltm.save(MemoryEntry(session_id="s1", entry_type="fact", content="pytest fixtures"))

        results = ltm.search("GIL")
        assert len(results) == 1
        assert "GIL" in results[0].content
        ltm.close()

    def test_search_filtered_by_type(self):
        from llm_eval.memory import LongTermMemory, MemoryEntry

        ltm = LongTermMemory()
        ltm.save(MemoryEntry(session_id="s1", entry_type="fact", content="GIL fact"))
        ltm.save(MemoryEntry(session_id="s1", entry_type="eval_result", content="GIL eval"))

        only_facts = ltm.search("GIL", entry_type="fact")
        assert len(only_facts) == 1
        assert only_facts[0].entry_type == "fact"
        ltm.close()

    def test_get_all_by_type(self):
        from llm_eval.memory import LongTermMemory, MemoryEntry

        ltm = LongTermMemory()
        for i in range(3):
            ltm.save(MemoryEntry(session_id="s1", entry_type="eval_result", content=f"eval {i}"))
        ltm.save(MemoryEntry(session_id="s1", entry_type="fact", content="a fact"))

        evals = ltm.get_all_by_type("eval_result")
        assert len(evals) == 3
        ltm.close()

    def test_metadata_roundtrip(self):
        from llm_eval.memory import LongTermMemory, MemoryEntry

        ltm = LongTermMemory()
        meta = {"scores": {"faithfulness": 0.9}, "model": "gpt-4o-mini"}
        entry = MemoryEntry(
            session_id="s1",
            entry_type="eval_result",
            content="some eval",
            metadata=meta,
        )
        ltm.save(entry)
        retrieved = ltm.get_all_by_type("eval_result")[0]
        assert retrieved.metadata["scores"]["faithfulness"] == pytest.approx(0.9)
        assert retrieved.metadata["model"] == "gpt-4o-mini"
        ltm.close()

    def test_ordering_newest_first(self):
        from llm_eval.memory import LongTermMemory, MemoryEntry

        ltm = LongTermMemory()
        for i in range(3):
            entry = MemoryEntry(
                session_id="s1",
                entry_type="fact",
                content=f"fact {i}",
                created_at=float(i),
            )
            ltm.save(entry)

        facts = ltm.get_all_by_type("fact")
        # Newest first → fact 2, fact 1, fact 0
        assert facts[0].content == "fact 2"
        ltm.close()


# ---------------------------------------------------------------------------
# MemoryStore (unified)
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_add_turn_stored_in_both_tiers(self):
        from llm_eval.memory import MemoryStore

        with MemoryStore() as store:
            store.add_turn("user", "What is the GIL?")
            store.add_turn("assistant", "The GIL is a mutex …")

            # Short-term
            recent = store.get_recent_turns()
            assert len(recent) == 2
            assert recent[0].role == "user"

            # Long-term
            history = store.get_session_history()
            assert len(history) == 2

    def test_add_eval_result_only_in_long_term(self):
        from llm_eval.memory import MemoryStore

        with MemoryStore() as store:
            store.add_eval_result(
                question="What is the GIL?",
                answer="The GIL is a mutex …",
                scores={"faithfulness": 0.9, "agentic": 0.8},
            )

            # Must NOT appear in the short-term conversation buffer
            assert len(store.get_recent_turns()) == 0

            # Must appear in long-term
            evals = store.get_eval_results()
            assert len(evals) == 1
            assert evals[0].metadata["scores"]["faithfulness"] == pytest.approx(0.9)

    def test_add_fact(self):
        from llm_eval.memory import MemoryStore

        with MemoryStore() as store:
            store.add_fact("Python 3.13 removes the GIL by default.", metadata={"source": "pep703"})
            facts = store.get_facts()
            assert len(facts) == 1
            assert facts[0].metadata["source"] == "pep703"

    def test_search_across_long_term(self):
        from llm_eval.memory import MemoryStore

        with MemoryStore() as store:
            store.add_turn("user", "Tell me about the GIL")
            store.add_fact("GIL prevents parallel threads in CPython")
            store.add_fact("asyncio is single-threaded")

            results = store.search("GIL")
            assert len(results) == 2

    def test_session_isolation(self):
        from llm_eval.memory import MemoryStore

        with MemoryStore(session_id="session-A") as store_a:
            store_a.add_turn("user", "Hello from A")

        with MemoryStore(session_id="session-B") as store_b:
            store_b.add_turn("user", "Hello from B")
            # Short-term is per-instance; long-term here is a separate :memory: DB
            assert len(store_b.get_recent_turns()) == 1
            assert store_b.get_recent_turns()[0].content == "Hello from B"

    def test_context_manager(self):
        from llm_eval.memory import MemoryStore

        with MemoryStore() as store:
            store.add_turn("user", "test message")
        # No exception should be raised when the context manager closes the connection.

    def test_short_term_eviction_does_not_affect_long_term(self):
        from llm_eval.memory import MemoryStore

        with MemoryStore(short_term_capacity=3) as store:
            for i in range(5):
                store.add_turn("user", f"message {i}")

            # Short-term capped at 3
            assert len(store.get_recent_turns()) == 3
            # Long-term has all 5
            assert len(store.get_session_history()) == 5


# ---------------------------------------------------------------------------
# RAGPipeline + memory integration
# ---------------------------------------------------------------------------


class TestRAGPipelineWithMemory:
    def test_turns_recorded_after_run(self):
        from llm_eval.memory import MemoryStore
        from llm_eval.rag_pipeline import RAGPipeline, SimpleRetriever

        with MemoryStore() as store:
            pipeline = RAGPipeline(retriever=SimpleRetriever(), memory=store)
            pipeline.run("What is the Python GIL?")

            # Both user question and assistant answer should be recorded.
            turns = store.get_session_history()
            assert len(turns) == 2
            roles = [t.role for t in turns]
            assert "user" in roles
            assert "assistant" in roles

    def test_pipeline_without_memory_still_works(self):
        from llm_eval.rag_pipeline import RAGPipeline, SimpleRetriever

        pipeline = RAGPipeline(retriever=SimpleRetriever())
        result = pipeline.run("What is the Python GIL?")
        assert result.answer  # non-empty

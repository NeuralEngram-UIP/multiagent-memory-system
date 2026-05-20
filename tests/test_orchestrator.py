# tests/test_memory_orchestrator.py

import pytest
import threading
from unittest.mock import MagicMock, call, patch
from memory.memory_orchestrator import (
    MemoryOrchestrator,
    EXPECTED_EMBEDDING_DIM,
)


# ─── Fixtures ──────────────────────────────────────────────

def make_embedding(dim=EXPECTED_EMBEDDING_DIM):
    return [0.1] * dim

def make_memory_obj(memory_id="mem-001", content="test content"):
    m = MagicMock()
    m.memory_id = memory_id
    m.content = content
    return m

def make_working():
    wm = MagicMock()
    wm.as_messages.return_value = []
    return wm

def make_episodic():
    em = MagicMock()
    em.add.return_value = "ep-001"
    em.grounded_retrieve.return_value = []
    em.prune_forgotten.return_value = []
    return em

def make_semantic():
    sm = MagicMock()
    sm.add.return_value = "sem-001"
    sm.search_multi.return_value = []
    return sm

def make_orchestrator():
    wm = make_working()
    em = make_episodic()
    sm = make_semantic()
    orc = MemoryOrchestrator(
        working_memory=wm,
        episodic_memory=em,
        semantic_memory=sm
    )
    return orc, wm, em, sm

EMBEDDING = make_embedding()


# ─── Init ──────────────────────────────────────────────────

class TestInit:

    def test_stores_all_tiers(self):
        orc, wm, em, sm = make_orchestrator()
        assert orc.working_memory is wm
        assert orc.episodic_memory is em
        assert orc.semantic_memory is sm

    def test_locks_are_created(self):
        orc, *_ = make_orchestrator()
        assert orc.working_lock is not None
        assert orc.episodic_lock is not None
        assert orc.semantic_lock is not None


# ─── _validate_embedding ──────────────────────────────────

class TestValidateEmbedding:

    def test_valid_embedding_passes(self):
        orc, *_ = make_orchestrator()
        orc._validate_embedding(EMBEDDING)  # no exception

    def test_not_a_list_raises(self):
        orc, *_ = make_orchestrator()
        with pytest.raises(ValueError, match="embedding must be a list"):
            orc._validate_embedding((0.1,) * EXPECTED_EMBEDDING_DIM)

    def test_empty_list_raises(self):
        orc, *_ = make_orchestrator()
        with pytest.raises(ValueError, match="embedding cannot be empty"):
            orc._validate_embedding([])

    def test_wrong_dimension_raises(self):
        orc, *_ = make_orchestrator()
        with pytest.raises(ValueError, match=str(EXPECTED_EMBEDDING_DIM)):
            orc._validate_embedding([0.1] * 10)

    def test_non_numeric_values_raise(self):
        orc, *_ = make_orchestrator()
        with pytest.raises(ValueError, match="numeric values"):
            orc._validate_embedding(["a"] * EXPECTED_EMBEDDING_DIM)

    def test_int_values_accepted(self):
        orc, *_ = make_orchestrator()
        orc._validate_embedding([1] * EXPECTED_EMBEDDING_DIM)  # no exception


# ─── _estimate_tokens ─────────────────────────────────────

class TestEstimateTokens:

    def test_empty_string_returns_zero(self):
        orc, *_ = make_orchestrator()
        assert orc._estimate_tokens("") == 0

    def test_whitespace_only_returns_zero(self):
        orc, *_ = make_orchestrator()
        assert orc._estimate_tokens("   ") == 0

    def test_single_word(self):
        orc, *_ = make_orchestrator()
        assert orc._estimate_tokens("hello") == int(1 * 1.3)

    def test_multiple_words(self):
        orc, *_ = make_orchestrator()
        assert orc._estimate_tokens("hello world foo") == int(3 * 1.3)

    def test_returns_int(self):
        orc, *_ = make_orchestrator()
        result = orc._estimate_tokens("some text here")
        assert isinstance(result, int)


# ─── store() ───────────────────────────────────────────────

class TestStore:

    def test_returns_all_ids(self):
        orc, *_ = make_orchestrator()
        result = orc.store(
            agent_id="agent_1",
            content="hello",
            embedding=EMBEDDING
        )
        assert "memory_id" in result
        assert "episodic_id" in result
        assert "semantic_id" in result

    def test_working_memory_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        wm.add.assert_called_once()

    def test_episodic_memory_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        em.add.assert_called_once()

    def test_semantic_memory_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        sm.add.assert_called_once()

    def test_default_role_is_user(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        _, kwargs = wm.add.call_args
        assert kwargs["role"] == "user"

    def test_custom_role_passed(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(
            agent_id="agent_1",
            content="hello",
            embedding=EMBEDDING,
            context={"role": "assistant"}
        )
        _, kwargs = wm.add.call_args
        assert kwargs["role"] == "assistant"

    def test_invalid_embedding_raises(self):
        orc, *_ = make_orchestrator()
        with pytest.raises(ValueError):
            orc.store(
                agent_id="agent_1",
                content="hello",
                embedding=[0.1] * 10
            )

    def test_episodic_id_in_semantic_metadata(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        _, kwargs = sm.add.call_args
        assert kwargs["metadata"]["episodic_id"] == "ep-001"

    def test_agent_id_in_semantic_metadata(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        _, kwargs = sm.add.call_args
        assert kwargs["metadata"]["agent_id"] == "agent_1"

    def test_confidence_defaults_to_one(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        _, kwargs = sm.add.call_args
        assert kwargs["metadata"]["confidence"] == 1.0

    def test_shared_defaults_to_false(self):
        orc, wm, em, sm = make_orchestrator()
        orc.store(agent_id="agent_1", content="hello", embedding=EMBEDDING)
        _, kwargs = sm.add.call_args
        assert kwargs["metadata"]["shared"] is False

    def test_rollback_on_semantic_failure(self):
        orc, wm, em, sm = make_orchestrator()
        sm.add.side_effect = RuntimeError("semantic down")
        with pytest.raises(RuntimeError):
            orc.store(
                agent_id="agent_1",
                content="hello",
                embedding=EMBEDDING
            )
        em.delete.assert_called_once_with("ep-001")

    def test_no_rollback_if_episodic_fails(self):
        """No episodic_id yet — rollback should not call delete."""
        orc, wm, em, sm = make_orchestrator()
        em.add.side_effect = RuntimeError("episodic down")
        with pytest.raises(RuntimeError):
            orc.store(
                agent_id="agent_1",
                content="hello",
                embedding=EMBEDDING
            )
        em.delete.assert_not_called()

    def test_unique_memory_ids_each_call(self):
        orc, *_ = make_orchestrator()
        r1 = orc.store(agent_id="a", content="x", embedding=EMBEDDING)
        r2 = orc.store(agent_id="a", content="y", embedding=EMBEDDING)
        assert r1["memory_id"] != r2["memory_id"]


# ─── retrieve() ────────────────────────────────────────────

class TestRetrieve:

    def test_returns_all_keys(self):
        orc, *_ = make_orchestrator()
        result = orc.retrieve(
            agent_id="agent_1",
            query="what?",
            embedding=EMBEDDING
        )
        assert set(result.keys()) == {
            "query", "working_memory",
            "episodic_memory", "semantic_memory", "fused_memory"
        }

    def test_query_in_result(self):
        orc, *_ = make_orchestrator()
        result = orc.retrieve(
            agent_id="agent_1",
            query="my query",
            embedding=EMBEDDING
        )
        assert result["query"] == "my query"

    def test_invalid_embedding_raises(self):
        orc, *_ = make_orchestrator()
        with pytest.raises(ValueError):
            orc.retrieve(
                agent_id="agent_1",
                query="q",
                embedding=[0.1] * 10
            )

    def test_semantic_search_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.retrieve(agent_id="agent_1", query="q", embedding=EMBEDDING)
        sm.search_multi.assert_called_once()

    def test_episodic_retrieve_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.retrieve(agent_id="agent_1", query="q", embedding=EMBEDDING)
        em.grounded_retrieve.assert_called_once()

    def test_working_context_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.retrieve(agent_id="agent_1", query="q", embedding=EMBEDDING)
        wm.as_messages.assert_called_once()

    def test_fused_memory_is_list(self):
        orc, *_ = make_orchestrator()
        result = orc.retrieve(
            agent_id="agent_1", query="q", embedding=EMBEDDING
        )
        assert isinstance(result["fused_memory"], list)


# ─── get_context() ─────────────────────────────────────────

class TestGetContext:

    def test_delegates_to_working_memory(self):
        orc, wm, em, sm = make_orchestrator()
        wm.as_messages.return_value = [{"role": "user", "content": "hi"}]
        result = orc.get_context(agent_id="agent_1", limit=5)
        wm.as_messages.assert_called_once_with(agent_id="agent_1", limit=5)
        assert result == [{"role": "user", "content": "hi"}]

    def test_default_limit_ten(self):
        orc, wm, *_ = make_orchestrator()
        orc.get_context(agent_id="agent_1")
        _, kwargs = wm.as_messages.call_args
        assert kwargs["limit"] == 10


# ─── reinforce() ───────────────────────────────────────────

class TestReinforce:

    def test_episodic_recall_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.reinforce(agent_id="agent_1", memory_id="mem-001")
        em.recall.assert_called_once_with(episode_id="mem-001")

    def test_semantic_reinforce_called(self):
        orc, wm, em, sm = make_orchestrator()
        orc.reinforce(agent_id="agent_1", memory_id="mem-001")
        sm.reinforce.assert_called_once_with("mem-001")

    def test_returns_none(self):
        orc, *_ = make_orchestrator()
        assert orc.reinforce(agent_id="a", memory_id="m") is None


# ─── cleanup() ─────────────────────────────────────────────

class TestCleanup:

    def test_returns_zero_when_nothing_removed(self):
        orc, *_ = make_orchestrator()
        assert orc.cleanup() == 0

    def test_returns_count_of_removed(self):
        orc, wm, em, sm = make_orchestrator()
        em.prune_forgotten.return_value = ["id1", "id2", "id3"]
        assert orc.cleanup() == 3

    def test_semantic_delete_called_per_id(self):
        orc, wm, em, sm = make_orchestrator()
        em.prune_forgotten.return_value = ["id1", "id2"]
        orc.cleanup()
        assert sm.delete.call_count == 2
        sm.delete.assert_any_call("id1")
        sm.delete.assert_any_call("id2")

    def test_invalid_prune_return_raises(self):
        orc, wm, em, sm = make_orchestrator()
        em.prune_forgotten.return_value = "not-a-list"
        with pytest.raises(TypeError, match="must return List"):
            orc.cleanup()

    def test_semantic_delete_failure_does_not_raise(self):
        """Semantic cleanup failures are swallowed per design."""
        orc, wm, em, sm = make_orchestrator()
        em.prune_forgotten.return_value = ["id1"]
        sm.delete.side_effect = Exception("db error")
        result = orc.cleanup()  # should not raise
        assert result == 1


# ─── apply_decay() ─────────────────────────────────────────

class TestApplyDecay:

    def test_delegates_to_episodic(self):
        orc, wm, em, sm = make_orchestrator()
        orc.apply_decay()
        em.apply_decay.assert_called_once()

    def test_returns_none(self):
        orc, *_ = make_orchestrator()
        assert orc.apply_decay() is None


# ─── _budgeted_fusion ──────────────────────────────────────

class TestBudgetedFusion:

    def test_empty_inputs_return_empty(self):
        orc, *_ = make_orchestrator()
        result = orc._budgeted_fusion([], [], 4000)
        assert result == []

    def test_deduplicates_by_memory_id(self):
        orc, *_ = make_orchestrator()
        m = make_memory_obj(memory_id="dup-id", content="hello")
        result = orc._budgeted_fusion([m], [m], 4000)
        assert len(result) == 1

    def test_deduplicates_by_content_when_no_id(self):
        orc, *_ = make_orchestrator()
        m1 = MagicMock(spec=["content"])
        m1.content = "same content"
        m2 = MagicMock(spec=["content"])
        m2.content = "same content"
        result = orc._budgeted_fusion([m1], [m2], 4000)
        assert len(result) == 1

    def test_token_budget_respected(self):
        orc, *_ = make_orchestrator()
        # each memory ~13 tokens (10 words * 1.3)
        memories = [
            make_memory_obj(memory_id=f"id-{i}", content="word " * 10)
            for i in range(20)
        ]
        result = orc._budgeted_fusion(memories, [], token_budget=50)
        total_tokens = sum(
            orc._estimate_tokens(m.content) for m in result
        )
        assert total_tokens <= 50

    def test_episodic_results_prioritised(self):
        """Episodic comes first in combined list."""
        orc, *_ = make_orchestrator()
        ep = make_memory_obj(memory_id="ep-1", content="episodic")
        sem = make_memory_obj(memory_id="sem-1", content="semantic")
        result = orc._budgeted_fusion([ep], [sem], 4000)
        assert result[0].memory_id == "ep-1"

    def test_breaks_on_budget_exceeded(self):
        """Once budget is exceeded, remaining memories are dropped."""
        orc, *_ = make_orchestrator()
        big = make_memory_obj(memory_id="big", content="word " * 400)
        small = make_memory_obj(memory_id="small", content="tiny")
        # big exceeds budget alone, small should never be added
        result = orc._budgeted_fusion([big, small], [], token_budget=100)
        assert len(result) == 0


# ─── Thread Safety ─────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_stores_no_exception(self):
        orc, *_ = make_orchestrator()
        errors = []

        def do_store():
            try:
                orc.store(
                    agent_id="agent_1",
                    content="concurrent content",
                    embedding=EMBEDDING
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_retrieves_no_exception(self):
        orc, *_ = make_orchestrator()
        errors = []

        def do_retrieve():
            try:
                orc.retrieve(
                    agent_id="agent_1",
                    query="q",
                    embedding=EMBEDDING
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_retrieve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
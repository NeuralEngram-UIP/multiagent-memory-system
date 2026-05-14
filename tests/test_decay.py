import pytest
from memory.memory_store import MemoryStore
from decay.decay_engine import DecayEngine

def test_retention_score_initial():
    store = MemoryStore()
    engine = DecayEngine(store)
    mem = store.add("Test memory", agent_id="test_agent")
    score = engine.compute_retention(mem["id"])
    assert 0.9 <= score <= 1.0

def test_reinforce_updates_score():
    store = MemoryStore()
    engine = DecayEngine(store)
    mem = store.add("Reinforcement test", agent_id="test_agent")
    engine.reinforce(mem["id"])
    updated = store.get(mem["id"])
    assert updated["access_count"] == 1

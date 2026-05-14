import pytest
from memory.memory_store import MemoryStore

def test_add_and_retrieve():
    store = MemoryStore()
    mem = store.add("Spaced repetition helps retention", agent_id="agent1", tags=["SRS"])
    assert mem["id"] in [m["id"] for m in store.all()]

def test_search_by_keyword():
    store = MemoryStore()
    store.add("Ebbinghaus curve paper", agent_id="agent1")
    store.add("Unrelated content", agent_id="agent1")
    results = store.search("Ebbinghaus")
    assert len(results) == 1

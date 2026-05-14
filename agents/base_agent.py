"""
Base Agent — defines the common interface for all agents in the system.
"""
from memory.memory_store import MemoryStore
from decay.decay_engine import DecayEngine

class BaseAgent:
    def __init__(self, agent_id: str, memory_store: MemoryStore, decay_engine: DecayEngine):
        self.agent_id = agent_id
        self.memory_store = memory_store
        self.decay_engine = decay_engine

    def retrieve(self, query: str):
        memories = self.memory_store.search(query, agent_id=self.agent_id)
        for mem in memories:
            self.decay_engine.reinforce(mem["id"])
        return memories

    def store(self, content: str, tags: list = []):
        return self.memory_store.add(content=content, agent_id=self.agent_id, tags=tags)

    def respond(self, prompt: str) -> str:
        raise NotImplementedError("Each agent must implement respond()")

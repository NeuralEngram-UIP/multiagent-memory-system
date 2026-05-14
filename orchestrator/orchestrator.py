"""
Orchestrator — coordinates task routing across multiple agents.
"""
from agents.researcher_agent import ResearcherAgent
from agents.synthesizer_agent import SynthesizerAgent
from memory.memory_store import MemoryStore
from decay.decay_engine import DecayEngine
from agents.executor_agent import ExecutorAgent

class Orchestrator:
    def __init__(self):
        self.memory_store = MemoryStore()
        self.decay_engine = DecayEngine(memory_store=self.memory_store)
        self.agents = {
            "researcher": ResearcherAgent(self.memory_store, self.decay_engine),
            "synthesizer": SynthesizerAgent(self.memory_store, self.decay_engine),
            "executor": ExecutorAgent(self.memory_store, self.decay_engine),

        }

    def route(self, task: str, agent_id: str = "researcher") -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return f"Agent '{agent_id}' not found."
        return agent.respond(task)

    def decay_pass(self):
        self.decay_engine.run_decay_pass()

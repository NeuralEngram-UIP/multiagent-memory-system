"""
Researcher Agent — responsible for gathering and storing new information.
"""
from agents.base_agent import BaseAgent

class ResearcherAgent(BaseAgent):
    def __init__(self, memory_store, decay_engine):
        super().__init__(agent_id="researcher", memory_store=memory_store, decay_engine=decay_engine)

    def respond(self, prompt: str) -> str:
        context = self.retrieve(prompt)
        return f"[ResearcherAgent] Retrieved {len(context)} memories for: {prompt}"

"""
Synthesizer Agent — consolidates insights from multiple agents' memory.
"""
from agents.base_agent import BaseAgent

class SynthesizerAgent(BaseAgent):
    def __init__(self, memory_store, decay_engine):
        super().__init__(agent_id="synthesizer", memory_store=memory_store, decay_engine=decay_engine)

    def respond(self, prompt: str) -> str:
        context = self.retrieve(prompt)
        return f"[SynthesizerAgent] Synthesizing {len(context)} memories for: {prompt}"

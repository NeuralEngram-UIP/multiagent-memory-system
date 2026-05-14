"""
Executor Agent — runs patient session, logs responses, updates memory strength.
"""
from agents.base_agent import BaseAgent

class ExecutorAgent(BaseAgent):
    def __init__(self, memory_store, decay_engine):
        super().__init__(agent_id="executor", memory_store=memory_store, decay_engine=decay_engine)

    def run_session(self, prompt: str) -> dict:
        """Run a session, retrieve context, log response."""
        context = self.retrieve(prompt)
        response = f"[ExecutorAgent] Processing: {prompt}"
        
        # Store the interaction as a new memory
        self.store(content=f"Session: {prompt} | Response: {response}", tags=["session", "log"])
        
        return {
            "prompt": prompt,
            "context_used": len(context),
            "response": response,
            "status": "logged"
        }

    def respond(self, prompt: str) -> str:
        result = self.run_session(prompt)
        return result["response"]
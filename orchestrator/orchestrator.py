"""
Orchestrator

Coordinates task routing
across agents using MemoryStore.
"""
import logging
from typing import Any, Callable, Dict, List

from sentence_transformers import SentenceTransformer
import anthropic

from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemoryStore
from memory.semantic_memory import SemanticMemoryStore
from memory.memory_orchestrator import MemoryOrchestrator
from memory.memory_store import MemoryStore
from memory.scheduler import MemoryScheduler
from agents.executor_agent import ExecutorAgent
from agents.memory_agent import MemoryAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordinates agents and memory stack.
    """

    def __init__(self):

        # --- embedding model ---
        self.embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # --- anthropic client ---
        self.client = anthropic.Anthropic()

        # --- memory stack ---
        working_memory = WorkingMemory(
            capacity=20
        )
        episodic_memory = EpisodicMemoryStore()
        semantic_memory = SemanticMemoryStore()

        memory_orchestrator = MemoryOrchestrator(
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory
        )

        self.memory_store = MemoryStore(
            orchestrator=memory_orchestrator
        )

        # --- scheduler ---
        self.scheduler = MemoryScheduler(
            memory_store=self.memory_store,
            interval_seconds=300
        )
        self.scheduler.start()

        # --- agents ---
        self.executor = ExecutorAgent(
            memory_store=self.memory_store,
            embed_fn=self._embed,
            llm_fn=self._llm
        )

        self.memory_agent = MemoryAgent(
            memory_store=self.memory_store
        )

        logger.info(
            "[Orchestrator] Initialized."
        )

    def _embed(
        self,
        text: str
    ) -> List[float]:
        """
        Generate embedding for text.
        """
        return (
            self.embed_model
            .encode(text)
            .tolist()
        )

    def _llm(
        self,
        task: str,
        context: List[Dict[str, str]]
    ) -> str:
        """
        Generate response via Claude.
        """
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=(
                "You are a helpful assistant "
                "with long-term memory. "
                "Use the provided context "
                "to answer accurately."
            ),
            messages=context + [
                {
                    "role": "user",
                    "content": task
                }
            ]
        )
        return response.content[0].text

    def route(
        self,
        task: str
    ) -> Dict[str, Any]:
        """
        Route task to executor agent.
        """
        logger.info(
            f"[Orchestrator] "
            f"Routing task: {task}"
        )
        return self.executor.execute(
            task=task
        )

    def stop(self):
        """
        Shutdown orchestrator cleanly.
        """
        self.scheduler.stop()
        logger.info(
            "[Orchestrator] Stopped."
        )
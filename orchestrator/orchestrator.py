# orchestrator.py

"""
Orchestrator

Coordinates task routing across agents.
Now routes through PlannerAgent
before ExecutorAgent.

Flow
────
User query
    ↓
Orchestrator.route()
    ↓
PlannerAgent.plan()
    ↓  (intent detection + optional memory retrieval)
ExecutorAgent.execute()
    ↓
Response
"""

import logging

from typing import Any, Dict, List

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
from agents.planner_agent import PlannerAgent     # ← Anika's agent


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates agents and memory stack.

    Agent pipeline
    ──────────────
    PlannerAgent  ← routes + pre-retrieves
    ExecutorAgent ← executes + stores
    MemoryAgent   ← lifecycle operations
    """

    def __init__(self):

        # ── Embedding model ──────────────────
        self.embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # ── Anthropic client ─────────────────
        self.client = anthropic.Anthropic()

        # ── Memory stack ─────────────────────
        working_memory  = WorkingMemory(capacity=20)
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

        # ── Scheduler ────────────────────────
        self.scheduler = MemoryScheduler(
            memory_store=self.memory_store,
            interval_seconds=300
        )
        self.scheduler.start()

        # ── Agents ───────────────────────────

        # ExecutorAgent first (PlannerAgent wraps it)
        self.executor = ExecutorAgent(
            memory_store=self.memory_store,
            embed_fn=self._embed,
            llm_fn=self._llm
        )

        self.memory_agent = MemoryAgent(
            memory_store=self.memory_store
        )

        # PlannerAgent wraps ExecutorAgent
        self.planner = PlannerAgent(
            memory_store=self.memory_store,
            executor=self.executor,
            embed_fn=self._embed
        )

        logger.info(
            "[Orchestrator] Initialized. "
            "Pipeline: PlannerAgent → ExecutorAgent"
        )

    # ─────────────────────────────────────────────────────────
    # Embed
    # ─────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────
    # LLM
    # ─────────────────────────────────────────────────────────

    def _llm(
        self,
        task: str,
        context: List[Dict[str, str]]
    ) -> str:
        """
        Generate response via Claude.
        """

        # Filter out system messages from context
        # to pass them properly to the API
        system_parts = [
            m["content"]
            for m in context
            if m["role"] == "system"
        ]

        non_system = [
            m for m in context
            if m["role"] != "system"
        ]

        system_prompt = (
            "You are a helpful assistant "
            "with long-term memory. "
            "Use the provided context "
            "to answer accurately.\n\n"
            + "\n\n".join(system_parts)
        ).strip()

        messages = non_system + [
            {
                "role": "user",
                "content": task
            }
        ]

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            messages=messages
        )

        return response.content[0].text

    # ─────────────────────────────────────────────────────────
    # Route — now goes through PlannerAgent
    # ─────────────────────────────────────────────────────────

    def route(
        self,
        task: str
    ) -> Dict[str, Any]:
        """
        Route task through PlannerAgent.

        PlannerAgent will:
        - detect intent
        - decide if memory needed
        - retrieve if needed
        - hand off to ExecutorAgent
        """

        logger.info(
            "[Orchestrator] Routing task: %r",
            task[:60]
        )

        return self.planner.plan(task=task)

    # ─────────────────────────────────────────────────────────
    # Stop
    # ─────────────────────────────────────────────────────────

    def stop(self):
        """
        Shutdown orchestrator cleanly.
        """

        self.scheduler.stop()

        logger.info(
            "[Orchestrator] Stopped."
        )
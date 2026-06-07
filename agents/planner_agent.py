# planner_agent.py

"""
Planner Agent — NeuralEngram

Responsibilities
────────────────
- task analysis (memory-needed or not)
- task decomposition into subtasks
- routing decision (memory → executor)
- memory retrieval orchestration
- response coordination

Architecture
────────────
User Query
    ↓
PlannerAgent.plan()
    ↓ (if memory needed)
BaseAgent.retrieve_memories()
    ↓
ExecutorAgent.execute()
    ↓
Response
"""

import logging
import re

from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.executor_agent import ExecutorAgent

from memory.memory_store import MemoryStore


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Intent Detection Signals
# ─────────────────────────────────────────────────────────────

MEMORY_SIGNALS = [
    "remember", "recall", "what did",
    "what do you know", "based on",
    "from before", "earlier", "last time",
    "you said", "we discussed", "history",
    "previous", "forgot", "remind me",
    "what was", "tell me about my",
]

REASONING_SIGNALS = [
    "analyze", "compare", "explain why",
    "evaluate", "reason", "pros and cons",
    "difference between", "how does",
    "why is", "contrast", "assess",
    "break down", "justify",
]

GENERAL_SIGNALS = [
    "what is", "define", "who is",
    "when was", "how do i", "how to",
    "give me", "list", "show me",
    "tell me", "describe",
]


# ─────────────────────────────────────────────────────────────
# Task Intent
# ─────────────────────────────────────────────────────────────

class TaskIntent:
    MEMORY    = "memory_retrieval"
    REASONING = "reasoning"
    GENERAL   = "general_response"


# ─────────────────────────────────────────────────────────────
# Planner Agent
# ─────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Cognitive task planner.

    PlannerAgent sits between the
    Orchestrator and ExecutorAgent.

    It decides:
    - whether memory retrieval is needed
    - what the task intent is
    - how to route the task

    This preserves ExecutorAgent's
    existing interface while adding
    intelligent pre-planning.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        executor: ExecutorAgent,
        embed_fn: Callable[[str], List[float]],
        agent_id: str = "planner_agent"
    ):

        super().__init__(
            agent_id=agent_id,
            memory_store=memory_store
        )

        self.executor  = executor
        self.embed_fn  = embed_fn

        logger.info(
            "PlannerAgent initialized "
            "agent_id=%s",
            self.agent_id
        )

    # ─────────────────────────────────────────────────────────
    # Intent Detection
    # ─────────────────────────────────────────────────────────

    def _detect_intent(
        self,
        task: str
    ) -> str:
        """
        Score task against signal lists
        and return the best-fit intent.

        Priority order:
        MEMORY > REASONING > GENERAL
        """

        lowered = task.lower()

        def score(signals):
            return sum(
                1 for s in signals
                if s in lowered
            )

        memory_score    = score(MEMORY_SIGNALS)
        reasoning_score = score(REASONING_SIGNALS)

        if memory_score > 0:
            return TaskIntent.MEMORY

        if reasoning_score > 0:
            return TaskIntent.REASONING

        return TaskIntent.GENERAL

    # ─────────────────────────────────────────────────────────
    # Memory Needed Check
    # ─────────────────────────────────────────────────────────

    def _requires_memory(
        self,
        intent: str
    ) -> bool:
        return True
        """
        Determine if memory retrieval
        should precede execution.

        GENERAL tasks skip retrieval
        to avoid unnecessary latency.
        """

      
    # ─────────────────────────────────────────────────────────
    # Plan
    # ─────────────────────────────────────────────────────────

    def plan(
        self,
        task: str,
        agent_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full planning pipeline.

        Steps
        ─────
        1. Detect intent
        2. Decide if memory needed
        3. If needed — retrieve & log
        4. Route to ExecutorAgent
        5. Reinforce used memories
        6. Return enriched result

        context:
            Metadata for memory storage.
            Passed through to executor.
        """

        task = task.strip()

        if not task:
            raise ValueError(
                "task cannot be empty"
            )

        # ── Phase 1: Intent ──────────────────

        intent = self._detect_intent(task)
        needs_memory = self._requires_memory(intent)

        logger.info(
            "[PlannerAgent] "
            "intent=%s | needs_memory=%s | "
            "task=%r",
            intent,
            needs_memory,
            task[:60]
        )

        # ── Phase 2: Pre-retrieval ───────────

        planner_memories = {}

        if needs_memory:

            embedding = self.embed_fn(task)

            planner_memories = self.retrieve_memories(
                query=task,
                embedding=embedding,
                agent_id=agent_id,
                top_k=5
            )

            fused = planner_memories.get(
                "fused_memory", []
            )

            logger.info(
                "[PlannerAgent] "
                "Pre-retrieval: fused=%d memories",
                len(fused)
            )

            # Reinforce retrieved memories

            for mem in fused:
                print(f">>> MEM TYPE: {type(mem)} attrs: {dir(mem)}")
                try:
                    self.reinforce_memory(
                     memory_id=mem.episode_id  # ← change mem.id to mem.episode_id
                 )
                except Exception as e:
                    logger.warning(
                  "[PlannerAgent] "
                 "Reinforce failed: %s", e
             )

        # ── Phase 3: Execute ─────────────────

        result = self.executor.execute(
            task=task,
            agent_id=agent_id,
            context=context
        )

        # ── Phase 4: Enrich result ───────────

        result["intent"]          = intent
        result["needs_memory"]    = needs_memory
        result["planner_memories"] = planner_memories

        logger.info(
            "[PlannerAgent] "
            "Task complete | intent=%s | "
            "response_len=%d",
            intent,
            len(result.get("response", ""))
        )

        return result
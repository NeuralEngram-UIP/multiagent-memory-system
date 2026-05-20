"""
FastAPI entry point for the
multi-agent memory system.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemoryStore
from memory.semantic_memory import SemanticMemoryStore
from memory.memory_orchestrator import MemoryOrchestrator
from memory.memory_store import MemoryStore
from memory.scheduler import MemoryScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MultiAgent Memory System",
    version="1.0.0"
)

# --- build memory stack ---
working_memory = WorkingMemory(capacity=20)
episodic_memory = EpisodicMemoryStore()
semantic_memory = SemanticMemoryStore()

orchestrator = MemoryOrchestrator(
    working_memory=working_memory,
    episodic_memory=episodic_memory,
    semantic_memory=semantic_memory
)

memory_store = MemoryStore(
    orchestrator=orchestrator
)

scheduler = MemoryScheduler(
    memory_store=memory_store,
    interval_seconds=300
)

scheduler.start()


# --- request models ---
class StoreRequest(BaseModel):
    content: str
    embedding: List[float]
    context: Optional[Dict[str, Any]] = None


class RetrieveRequest(BaseModel):
    query: str
    embedding: List[float]
    top_k: int = 5


class ReinforceRequest(BaseModel):
    memory_id: str


# --- routes ---
@app.get("/")
def root():
    return {
        "status": "running",
        "system": "MultiAgent Memory System"
    }


@app.post("/store")
def store_memory(req: StoreRequest):
    """
    Store memory across all tiers.
    """
    memory_ids = memory_store.store(
        content=req.content,
        embedding=req.embedding,
        context=req.context
    )
    return {
        "status": "stored",
        "memory_ids": memory_ids
    }


@app.post("/retrieve")
def retrieve_memories(req: RetrieveRequest):
    """
    Retrieve relevant memories.
    """
    results = memory_store.retrieve(
        query=req.query,
        embedding=req.embedding,
        top_k=req.top_k
    )
    return results


@app.get("/context")
def get_context(limit: int = 10):
    """
    Retrieve working memory context.
    """
    return memory_store.get_context(
        limit=limit
    )


@app.post("/reinforce")
def reinforce_memory(req: ReinforceRequest):
    """
    Reinforce a recalled memory.
    """
    memory_store.reinforce(
        memory_id=req.memory_id
    )
    return {
        "status": "reinforced",
        "memory_id": req.memory_id
    }


@app.post("/decay")
def trigger_decay():
    """
    Manually trigger decay pass.
    """
    memory_store.apply_decay()
    return {
        "status": "decay applied"
    }


@app.post("/cleanup")
def cleanup_memories():
    """
    Remove forgotten memories.
    """
    removed = memory_store.cleanup()
    return {
        "status": "cleanup complete",
        "removed": removed
    }


@app.get("/scheduler/status")
def scheduler_status():
    """
    Check scheduler running state.
    """
    return {
        "running": scheduler.is_running()
    }


@app.on_event("shutdown")
def shutdown():
    scheduler.stop()
    logger.info(
        "Scheduler stopped."
    )
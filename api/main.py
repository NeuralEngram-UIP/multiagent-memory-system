"""
FastAPI entry point for the
multi-agent memory system.
"""
import logging
from typing import Any, Dict, List, Optional
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import FileResponse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from orchestrator.orchestrator import Orchestrator
from memory.scheduler import MemoryScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MultiAgent Memory System",
    version="1.0.0"
)

# ── CORS — required for browser to call API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Single orchestrator — owns the memory stack ──
orch = Orchestrator()


# ── Request models ──
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class StoreRequest(BaseModel):
    agent_id: str = "default"
    content: str
    embedding: List[float]
    context: Optional[Dict[str, Any]] = None


class RetrieveRequest(BaseModel):
    agent_id: str = "default"
    query: str
    embedding: List[float]
    top_k: int = 5


class ReinforceRequest(BaseModel):
    agent_id: str = "default"
    memory_id: str


# ── Routes ──

@app.get("/")
def root():
    return {
        "status": "running",
        "system": "MultiAgent Memory System"
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Main chat route — routes through
    PlannerAgent → ExecutorAgent.
    """
    result = orch.route(task=req.message,
    agent_id=req.user_id ) 
    fused = result.get(
        "retrieved_memories", {}
    ).get("fused_memory", [])
    return {
        "reply": result["response"],
        "agent_used": "executor",
        "memories_retrieved": len(fused),
        "routing_path": ["planner", "executor"]
    }


# ── KEY ROUTE — live memory panel ──
@app.get("/metrics/{agent_id}")
def get_metrics(agent_id: str):
    """
    Live retention, stability, priority
    scores for the memory panel.
    """
    try:
        result = (
            orch.memory_store
            .orchestrator
            .metrics(agent_id=agent_id)
        )
        return result
    except Exception as e:
        logger.error("Metrics error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/memories/{agent_id}")
def get_memories(agent_id: str):
    try:
        result = (
            orch.memory_store
            .orchestrator
            .metrics(agent_id=agent_id)
        )
        return {"episodes": result["memories"]}
    except Exception as e:
        logger.error("Memories error: %s", e)
        return {"episodes": []}


@app.post("/store")
def store_memory(req: StoreRequest):
    memory_ids = orch.memory_store.store(
        agent_id=req.agent_id,
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
    results = orch.memory_store.retrieve(
        agent_id=req.agent_id,
        query=req.query,
        embedding=req.embedding,
        top_k=req.top_k
    )
    return results


@app.get("/context/{agent_id}")
def get_context(agent_id: str, limit: int = 10):
    return orch.memory_store.get_context(
        agent_id=agent_id,
        limit=limit
    )


@app.post("/reinforce")
def reinforce_memory(req: ReinforceRequest):
    orch.memory_store.reinforce(
        agent_id=req.agent_id,
        memory_id=req.memory_id
    )
    return {
        "status": "reinforced",
        "memory_id": req.memory_id
    }


@app.post("/decay")
def trigger_decay():
    orch.memory_store.apply_decay()
    return {"status": "decay applied"}


@app.post("/cleanup")
def cleanup_memories():
    removed = orch.memory_store.cleanup()
    return {
        "status": "cleanup complete",
        "removed": removed
    }


@app.get("/scheduler/status")
def scheduler_status():
    return {
        "running": orch.scheduler.is_running()
    }


@app.get("/ui")
def serve_ui():
    html_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "maem_multiagent.html"
    )
    return FileResponse(html_path)


@app.on_event("shutdown")
def shutdown():
    orch.stop()
    logger.info("Scheduler stopped.")

class SimulateTimeRequest(BaseModel):
    agent_id: str
    hours: float = 12.0

@app.post("/simulate_time")
def simulate_time(req: SimulateTimeRequest):
    try:
        with orch.memory_store.orchestrator.episodic_lock:
            episodes = orch.memory_store.orchestrator.episodic_memory.get_all()
            print(f">>> SIMULATE: agent={req.agent_id} hours={req.hours} total={len(episodes)}")
            for ep in episodes:
                if ep.agent_id == req.agent_id:
                    ep.last_reviewed_at = (
                        ep.last_reviewed_at
                        - timedelta(hours=req.hours)
                    )
                    orch.memory_store.orchestrator.episodic_memory._save_episode(ep)
                    print(f">>> AGED: {ep.content[:30]}")
        return {"status": "ok", "hours_aged": req.hours}
    except Exception as e:
        print(f">>> SIMULATE ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/alerts/{agent_id}")
def get_alerts(agent_id: str):
    """
    Get caregiver alerts for agent.
    """
    try:
        with orch.memory_store.orchestrator.episodic_lock:
            episodes = orch.memory_store.orchestrator.episodic_memory.get_all()
        
        CRITICAL_KEYWORDS = [
            "medicine", "medication", "doctor",
            "appointment", "emergency", "hospital",
            "blood pressure", "insulin", "tablet",
            "daughter", "son", "family", "home"
        ]
        
        alerts = []
        for ep in episodes:
            if ep.agent_id != agent_id:
                continue
            retention = ep.retention()
            content_lower = str(ep.content).lower()
            is_critical = any(
                kw in content_lower
                for kw in CRITICAL_KEYWORDS
            )
            if is_critical and retention < 0.5:
                alerts.append({
                    "memory": str(ep.content)[:80],
                    "retention": round(retention, 3),
                    "stability_hours": round(ep.stability_hours, 1),
                    "alert_level": "critical" if retention < 0.3 else "warning"
                })
        
        return {
            "agent_id": agent_id,
            "alert_count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error("Alerts error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    
class EvalRequest(BaseModel):
    agent_id: str = "alice"
    test_questions: List[str] = []

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    try:
        results = []
        for question in req.test_questions:
            # MAEM response (with memory)
            maem_result = orch.route(
                task=question,
                agent_id=req.agent_id
            )
            maem_response = maem_result["response"]
            memories_used = len(
                maem_result.get("retrieved_memories", {})
                .get("fused_memory", [])
            )

            # Baseline response (no memory — empty context)
            baseline_response = orch.planner.executor.llm_fn(
                question,
                []  # ← empty context, no memories
            )

            # Score baseline — did it mention anything specific?
            # If response contains "don't know", "no information",
            # "can't recall" — it failed to recall
            baseline_failed_keywords = [
                "don't have", "no information",
                "don't know", "cannot", "no context",
                "no memory", "no previous", "no stored",
                "haven't", "not aware", "no access"
            ]
            baseline_recalled = not any(
                kw in baseline_response.lower()
                for kw in baseline_failed_keywords
            )

            # Score MAEM — did it use memories?
            maem_recalled = memories_used > 0

            results.append({
                "question": question,
                "maem_response": maem_response[:200],
                "baseline_response": baseline_response[:200],
                "memories_used": memories_used,
                "maem_used_memory": maem_recalled,
                "baseline_recalled": baseline_recalled
            })

        total = len(results)
        maem_recall = sum(1 for r in results if r["maem_used_memory"])
        baseline_recall = sum(1 for r in results if r["baseline_recalled"])

        return {
            "total_questions": total,
            "maem_recall_rate": round(maem_recall / total, 3) if total > 0 else 0,
            "baseline_recall_rate": round(baseline_recall / total, 3) if total > 0 else 0,
            "improvement": round((maem_recall - baseline_recall) / total, 3) if total > 0 else 0,
            "results": results
        }
    except Exception as e:
        logger.error("Eval error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
"""
FastAPI entry point for the multi-agent memory system.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator.orchestrator import Orchestrator

app = FastAPI(title="MultiAgent Memory System", version="1.0.0")
orchestrator = Orchestrator()

class TaskRequest(BaseModel):
    task: str
    agent_id: str = "researcher"

@app.get("/")
def root():
    return {"status": "running", "system": "MultiAgent Memory System UIP17"}

@app.post("/task")
def run_task(req: TaskRequest):
    result = orchestrator.route(task=req.task, agent_id=req.agent_id)
    return {"agent": req.agent_id, "result": result}

@app.post("/decay")
def trigger_decay():
    orchestrator.decay_pass()
    return {"status": "decay pass complete"}

@app.get("/memories")
def list_memories():
    return orchestrator.memory_store.all()

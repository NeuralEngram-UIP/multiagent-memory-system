"""
Memory Store — handles storage, indexing, and retrieval of memory units.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Optional

class MemoryStore:
    def __init__(self):
        self._store: Dict[str, dict] = {}

    def add(self, content: str, agent_id: str, tags: List[str] = []) -> dict:
        mem_id = str(uuid.uuid4())
        memory = {
            "id": mem_id,
            "content": content,
            "agent_id": agent_id,
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": 0,
            "retention_score": 1.0
        }
        self._store[mem_id] = memory
        return memory

    def get(self, mem_id: str) -> Optional[dict]:
        return self._store.get(mem_id)

    def search(self, query: str, agent_id: Optional[str] = None) -> List[dict]:
        results = []
        for mem in self._store.values():
            if agent_id and mem["agent_id"] != agent_id:
                continue
            if query.lower() in mem["content"].lower():
                results.append(mem)
        return sorted(results, key=lambda m: m["retention_score"], reverse=True)

    def update_retention(self, mem_id: str, score: float):
        if mem_id in self._store:
            self._store[mem_id]["retention_score"] = score
            self._store[mem_id]["last_accessed"] = datetime.utcnow().isoformat()
            self._store[mem_id]["access_count"] += 1

    def all(self) -> List[dict]:
        return list(self._store.values())

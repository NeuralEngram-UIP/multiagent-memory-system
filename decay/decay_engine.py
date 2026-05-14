"""
Decay Engine — implements Ebbinghaus's forgetting curve.
R(t) = e^(-t / S)
"""
import math
from datetime import datetime
from memory.memory_store import MemoryStore

class DecayEngine:
    def __init__(self, memory_store: MemoryStore, stability: float = 1.0, decay_rate: float = 0.5):
        self.memory_store = memory_store
        self.stability = stability
        self.decay_rate = decay_rate

    def compute_retention(self, mem_id: str) -> float:
        mem = self.memory_store.get(mem_id)
        if not mem:
            return 0.0
        last = datetime.fromisoformat(mem["last_accessed"])
        t = (datetime.utcnow() - last).total_seconds() / 3600
        score = math.exp(-self.decay_rate * t / self.stability)
        return round(score, 4)

    def reinforce(self, mem_id: str):
        mem = self.memory_store.get(mem_id)
        if not mem:
            return
        self.stability *= 1.2
        new_score = self.compute_retention(mem_id)
        self.memory_store.update_retention(mem_id, new_score)

    def run_decay_pass(self):
        for mem in self.memory_store.all():
            score = self.compute_retention(mem["id"])
            self.memory_store.update_retention(mem["id"], score)

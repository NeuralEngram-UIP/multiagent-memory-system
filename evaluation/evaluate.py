"""
Evaluation script — measures retention accuracy and decay behavior.
"""
from orchestrator.orchestrator import Orchestrator

def run_evaluation():
    orch = Orchestrator()
    orch.agents["researcher"].store("Ebbinghaus forgetting curve foundational paper", tags=["memory", "theory"])
    orch.agents["researcher"].store("MemGPT architecture for long-term memory", tags=["AI", "memory"])
    orch.agents["researcher"].store("SM-2 spaced repetition algorithm", tags=["SRS", "memory"])

    print("=== Initial Memory State ===")
    for mem in orch.memory_store.all():
        print(f"  [{mem['id'][:8]}] score={mem['retention_score']} | {mem['content'][:50]}")

    print("\n=== Running Decay Pass ===")
    orch.decay_pass()

    print("\n=== Post-Decay Memory State ===")
    for mem in orch.memory_store.all():
        print(f"  [{mem['id'][:8]}] score={mem['retention_score']} | {mem['content'][:50]}")

if __name__ == "__main__":
    run_evaluation()

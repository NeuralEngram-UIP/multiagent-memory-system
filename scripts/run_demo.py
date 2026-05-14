"""
Demo Session Runner — runs a live session through the ExecutorAgent.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.orchestrator import Orchestrator

def run_demo():
    print("=== MultiAgent Memory System — Demo Session ===\n")
    
    orch = Orchestrator()

    # Seed some memories first
    orch.agents["executor"].store("Ebbinghaus forgetting curve reduces retention over time", tags=["theory"])
    orch.agents["executor"].store("Spaced repetition reinforces memory strength", tags=["theory"])
    orch.agents["executor"].store("MemGPT uses external memory for long context", tags=["AI"])

    print("✅ Memory seeded\n")

    # Run sessions
    sessions = [
        "What do we know about memory retention?",
        "How does spaced repetition work?",
        "Explain MemGPT memory architecture",
    ]

    for prompt in sessions:
        print(f"🧠 Prompt: {prompt}")
        result = orch.agents["executor"].run_session(prompt)
        print(f"   Context used: {result['context_used']} memories")
        print(f"   Response: {result['response']}")
        print(f"   Status: {result['status']}\n")

    # Show memory state after sessions
    print("=== Memory State After Sessions ===")
    for mem in orch.memory_store.all():
        print(f"  [{mem['id'][:8]}] score={mem['retention_score']} | access={mem['access_count']} | {mem['content'][:60]}")

if __name__ == "__main__":
    run_demo()
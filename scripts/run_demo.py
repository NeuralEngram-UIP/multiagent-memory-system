"""
Demo Session Runner — runs a live session through the ExecutorAgent.
"""
import sys
import os
import time
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
    print("⏳ Simulating time passing (5 seconds)...")
    time.sleep(5)

    # Run decay pass to drop scores
    # First decay pass
    orch.decay_engine.run_decay_pass(simulate_hours=6)
    print("📉 Decay pass complete (simulated 6hrs)\n")
    
    # Run sessions
    sessions = [
        ("What do we know about Ebbinghaus forgetting curve?", "executor"),
        ("How does spaced repetition strengthen memory?", "executor"),
        ("Explain MemGPT external memory for long context", "executor"),
    ]

    for prompt, agent_id in sessions:
        print(f"🧠 Prompt: {prompt}")
        result = orch.agents[agent_id].run_session(prompt)
        print(f"   Context used: {result['context_used']} memories")
        print(f"   Response: {result['response']}")
        print(f"   Status: {result['status']}\n")

    # Run another decay pass
    orch.decay_engine.run_decay_pass(simulate_hours=12)

    # Show memory state after sessions
    print("=== Memory State After Sessions ===")
    for mem in orch.memory_store.all():
        print(f"  [{mem['id'][:8]}] score={mem['retention_score']} | access={mem['access_count']} | {mem['content'][:60]}")

if __name__ == "__main__":
    run_demo()
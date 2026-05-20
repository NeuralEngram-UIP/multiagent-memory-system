"""
Evaluation Script

Measures retention accuracy
and decay behavior against
the real memory stack.
"""

from sentence_transformers import (
    SentenceTransformer
)

from memory.working_memory import (
    WorkingMemory
)

from memory.episodic_memory import (
    EpisodicMemoryStore
)

from memory.semantic_memory import (
    SemanticMemoryStore
)

from memory.memory_orchestrator import (
    MemoryOrchestrator
)

from memory.memory_store import (
    MemoryStore
)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

AGENT_ID = "eval_agent"


# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

def build_memory_stack():
    """
    Lazily initialize the full
    memory stack.

    IMPORTANT
    ─────────
    Initialization is intentionally
    deferred until runtime to avoid:

    - heavy imports during testing
    - unnecessary model loading
    - accidental DB initialization
    - Qdrant startup during import
    """

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    def embed(
        text: str
    ):

        return embed_model.encode(
            text
        ).tolist()

    working_memory = WorkingMemory(
        capacity=20
    )

    episodic_memory = (
        EpisodicMemoryStore()
    )

    semantic_memory = (
        SemanticMemoryStore()
    )

    orchestrator = (
        MemoryOrchestrator(
            working_memory=(
                working_memory
            ),

            episodic_memory=(
                episodic_memory
            ),

            semantic_memory=(
                semantic_memory
            )
        )
    )

    memory_store = MemoryStore(
        orchestrator=orchestrator
    )

    return {
        "embed": embed,
        "working_memory": (
            working_memory
        ),
        "episodic_memory": (
            episodic_memory
        ),
        "semantic_memory": (
            semantic_memory
        ),
        "memory_store": (
            memory_store
        )
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def print_episodic_state(
    episodic_memory
):

    print(
        "\n=== Episodic Memory State ==="
    )

    if hasattr(
        episodic_memory,
        "get_by_retention"
    ):

        episodes = (
            episodic_memory
            .get_by_retention(
                limit=10
            )
        )

    elif hasattr(
        episodic_memory,
        "all"
    ):

        episodes = (
            episodic_memory
            .all()
        )[:10]

    else:

        print(
            "No compatible episodic "
            "retrieval method found."
        )

        return

    for ep in episodes:

        retention = (
            ep.retention()
            if hasattr(
                ep,
                "retention"
            )
            else 0.0
        )

        stability = getattr(
            ep,
            "stability_hours",
            0.0
        )

        print(
            f"  [{ep.episode_id[:8]}] "
            f"retention={retention:.4f} | "
            f"stability={stability:.2f} | "
            f"{ep.content[:60]}"
        )


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def run_evaluation():

    stack = build_memory_stack()

    embed = stack["embed"]

    episodic_memory = (
        stack["episodic_memory"]
    )

    memory_store = (
        stack["memory_store"]
    )

    print(
        "\n=================================="
    )

    print(
        " MEMORY SYSTEM EVALUATION "
    )

    print(
        "=================================="
    )

    # ─────────────────────────────────────────
    # Store Memories
    # ─────────────────────────────────────────

    print(
        "\n=== Storing Memories ==="
    )

    ids1 = memory_store.store(
        agent_id=AGENT_ID,

        content=(
            "Ebbinghaus forgetting "
            "curve foundational paper"
        ),

        embedding=embed(
            "Ebbinghaus forgetting "
            "curve foundational paper"
        ),

        context={
            "role": "user"
        }
    )

    print(
        f"Stored: {ids1}"
    )

    ids2 = memory_store.store(
        agent_id=AGENT_ID,

        content=(
            "MemGPT architecture "
            "for long-term memory"
        ),

        embedding=embed(
            "MemGPT architecture "
            "for long-term memory"
        ),

        context={
            "role": "user"
        }
    )

    print(
        f"Stored: {ids2}"
    )

    ids3 = memory_store.store(
        agent_id=AGENT_ID,

        content=(
            "SM-2 spaced repetition "
            "algorithm"
        ),

        embedding=embed(
            "SM-2 spaced repetition "
            "algorithm"
        ),

        context={
            "role": "user"
        }
    )

    print(
        f"Stored: {ids3}"
    )

    # ─────────────────────────────────────────
    # Initial State
    # ─────────────────────────────────────────

    print_episodic_state(
        episodic_memory
    )

    # ─────────────────────────────────────────
    # Apply Decay
    # ─────────────────────────────────────────

    print(
        "\n=== Applying Decay ==="
    )

    memory_store.apply_decay()

    print(
        "Decay applied."
    )

    print_episodic_state(
        episodic_memory
    )

    # ─────────────────────────────────────────
    # Reinforcement
    # ─────────────────────────────────────────

    print(
        "\n=== Reinforcing First Memory ==="
    )

    memory_store.reinforce(
        agent_id=AGENT_ID,

        memory_id=(
            ids1["episodic_id"]
        )
    )

    print(
        f"Reinforced: "
        f"{ids1['episodic_id'][:8]}"
    )

    print_episodic_state(
        episodic_memory
    )

    # ─────────────────────────────────────────
    # Retrieval Test
    # ─────────────────────────────────────────

    print(
        "\n=== Semantic Retrieval ==="
    )

    results = memory_store.retrieve(
        agent_id=AGENT_ID,

        query="memory architecture",

        embedding=embed(
            "memory architecture"
        ),

        top_k=3
    )

    print(
        "\nSemantic Results:"
    )

    for mem in results.get(
        "semantic_memory",
        []
    ):

        print(
            f"  score={mem.score:.4f} | "
            f"{mem.content[:60]}"
        )

    # ─────────────────────────────────────────
    # Fused Memory
    # ─────────────────────────────────────────

    print(
        "\n=== Fused Retrieval ==="
    )

    fused = results.get(
        "fused_memory",
        []
    )

    for mem in fused:

        print(
            f"  fused | "
            f"{mem.content[:60]}"
        )

    # ─────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────

    print(
        "\n=== Cleanup Forgotten Memories ==="
    )

    removed = (
        memory_store.cleanup()
    )

    print(
        f"Removed "
        f"{removed} forgotten memories."
    )

    # ─────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────

    print(
        "\n=== Metrics ==="
    )

    metrics = (
        memory_store.metrics()
    )

    for key, value in (
        metrics.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        "\n=================================="
    )

    print(
        " EVALUATION COMPLETE "
    )

    print(
        "==================================\n"
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    run_evaluation()
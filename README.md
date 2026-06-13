# MultiAgent Ebbinghaus Memory Based Model

A multi-agent AI system with biologically-inspired memory management using Ebbinghaus's forgetting curve.

## Project Overview
This system implements cognitive memory decay and spaced reinforcement across a network of collaborative AI agents.

## Architecture
```
Orchestrator
    ├── Agent 1 ──┐
    ├── Agent 2 ──┼──► Shared Memory Store ──► Decay Engine
    └── Agent N ──┘
```

## Modules
| Module | Description |
|--------|-------------|
| `agents/` | Individual agent definitions and roles |
| `memory/` | Memory storage schema and retrieval logic |
| `decay/` | Ebbinghaus decay function and scoring |
| `orchestrator/` | Multi-agent coordination and routing |
| `evaluation/` | Testing, metrics, and evaluation scripts |
| `api/` | REST API layer for external interaction |

## 📌 Project Overview

This project implements a biologically-inspired multi-agent memory system designed around the principles of human cognitive forgetting and reinforcement. It is based on Ebbinghaus’s Forgetting Curve, which models how memory retention decreases over time unless actively reinforced.

The system consists of multiple collaborative AI agents that share a structured memory ecosystem. Each agent generates, retrieves, and updates memory entries while a centralized memory store manages persistence and decay. Over time, less frequently accessed information naturally fades, while important and repeatedly reinforced information becomes stronger and more persistent.

A dedicated decay engine continuously evaluates memory strength using time-based degradation and reinforcement signals. This ensures that the system does not behave like a static database, but rather like a dynamic cognitive system that evolves based on usage patterns.

The orchestrator coordinates communication between agents, ensuring controlled memory sharing, conflict resolution, and efficient routing of contextual information. This enables scalable multi-agent interaction while maintaining memory consistency and relevance.

Overall, the system is designed to simulate realistic memory behavior in AI systems, improving long-term reasoning, reducing noise, and enabling more human-like adaptive intelligence.

## ⚙️ How It Works

1. **Agent Interaction**  
   Each agent receives input, processes context, and generates memory entries based on its role and task.

2. **Memory Creation & Storage**  
   Important information is stored in the shared memory system with metadata such as timestamp, importance score, and usage frequency.

3. **Memory Scoring**  
   Each memory is assigned a dynamic score based on:
   - Recency of access  
   - Frequency of reinforcement  
   - Importance level assigned by agents  

4. **Decay Process**  
   The decay engine continuously applies the Ebbinghaus forgetting function to reduce memory strength over time for unused data.

5. **Reinforcement Cycle**  
   Frequently accessed memories are strengthened, slowing down their decay and increasing their retrieval priority.

6. **Retrieval & Coordination**  
   The orchestrator ensures agents retrieve the most relevant memories while maintaining consistency across the system.

7. **Optimization Loop**  
   Over time, the system self-optimizes by retaining only high-value memory entries and discarding low-relevance data.
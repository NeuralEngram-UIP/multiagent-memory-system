# multiagent-memory-system-UIP17

A multi-agent AI system with biologically-inspired memory management using Ebbinghaus's forgetting curve.

## Project Overview
This system implements cognitive memory decay and spaced reinforcement across a network of collaborative AI agents — part of Team TimeOut's Phase 1 project under UIP17.

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

## Setup
```bash
pip install -r requirements.txt
python scripts/run_system.py
```

## Team
Team TimeOut — UIP17

# Agentic AI Examples for PAI Oncology Trial Federated Learning

**Version:** 0.5.0
**Last Updated:** 2026-02-17
**License:** MIT

> **DISCLAIMER: RESEARCH USE ONLY.**
> This software is provided for research and educational purposes only.
> It has NOT been validated for clinical use, is NOT approved by the FDA
> or any other regulatory body, and MUST NOT be used to make clinical
> decisions or direct patient care.

## Overview

Six production-patterned agentic AI examples demonstrating how autonomous
agents integrate with the Physical AI Federated Learning oncology clinical
trials platform. Each script is self-contained, importable without GPU or
specialized hardware, and follows conditional-import conventions for
optional dependencies.

## Examples

| # | Script | Description | Key Concepts | Agent Framework |
|---|--------|-------------|--------------|-----------------|
| 01 | `01_mcp_oncology_server.py` | MCP server exposing oncology domain tools and resources for LLM agents | Tool definitions, resource URIs, JSON-RPC, 21 CFR Part 11 audit trail | MCP (Model Context Protocol) |
| 02 | `02_react_treatment_planner.py` | ReAct (Reason+Act) agent with thought/action/observation loops for treatment planning | Iterative reasoning, digital twin simulation, treatment recommendation | ReAct pattern, Anthropic / OpenAI |
| 03 | `03_realtime_adaptive_monitoring_agent.py` | Real-time adaptive agent with streaming multi-modal clinical data | Vital signs, lab results, imaging reports, cross-modal correlation, adaptive thresholds | Streaming agent, event-driven |
| 04 | `04_autonomous_simulation_orchestrator.py` | Multi-framework simulation orchestration agent | Isaac Sim, MuJoCo, PyBullet job management, result comparison, config optimization | Autonomous orchestrator |
| 05 | `05_safety_constrained_agent_executor.py` | Safety-gated agent with pre/post-conditions and human-in-the-loop | IEC 80601-2-77, ISO 14971, safety gates, emergency stop, rollback | Safety-constrained executor |
| 06 | `06_oncology_rag_compliance_agent.py` | RAG agent grounded in trial protocols, FDA guidance, ICH E6(R3), IEC standards | Retrieval-augmented generation, citation tracking, regulatory compliance | RAG pipeline |

## Quick Start

### Prerequisites

```bash
# Core dependency (required)
pip install numpy>=1.24.0

# Optional dependencies (scripts degrade gracefully without these)
pip install anthropic>=0.39.0      # Anthropic Claude API
pip install openai>=1.0.0          # OpenAI API
pip install mcp>=1.1.0             # Model Context Protocol SDK
pip install chromadb>=0.4.0        # Vector store for RAG
pip install scikit-learn>=1.3.0    # ML utilities
```

### Running Examples

Each script can be run directly or imported as a module:

```bash
# Run any example directly
python agentic-ai/examples-agentic-ai/01_mcp_oncology_server.py
python agentic-ai/examples-agentic-ai/02_react_treatment_planner.py
python agentic-ai/examples-agentic-ai/03_realtime_adaptive_monitoring_agent.py
python agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py
python agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py
python agentic-ai/examples-agentic-ai/06_oncology_rag_compliance_agent.py
```

```python
# Import specific components
from agentic_ai.examples_agentic_ai.01_mcp_oncology_server import OncologyMCPServer
from agentic_ai.examples_agentic_ai.02_react_treatment_planner import ReActTreatmentPlanner
```

### Linting

All scripts are validated against the project ruff configuration:

```bash
ruff check agentic-ai/examples-agentic-ai/
ruff format --check agentic-ai/examples-agentic-ai/
```

## Architecture

```
agentic-ai/examples-agentic-ai/
    01_mcp_oncology_server.py          # MCP tool/resource server
    02_react_treatment_planner.py      # ReAct reasoning loop
    03_realtime_adaptive_monitoring_agent.py  # Streaming monitor
    04_autonomous_simulation_orchestrator.py  # Multi-sim orchestrator
    05_safety_constrained_agent_executor.py   # Safety-gated executor
    06_oncology_rag_compliance_agent.py       # RAG compliance agent
    README.md                          # This file
```

## Conventions

- **`from __future__ import annotations`** in every file
- **`HAS_X` conditional imports** for all optional dependencies
- **`@dataclass`** for structured data with `field()` defaults
- **`Enum`** for categorical types
- **Structured logging** via `logging.getLogger(__name__)`
- **21 CFR Part 11** compatible audit trail patterns
- **RESEARCH USE ONLY** disclaimer in every module docstring

## References

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [ICH E6(R3) Good Clinical Practice](https://www.ich.org/)
- [FDA 21 CFR Part 11](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11)
- [IEC 80601-2-77 Medical Robotics](https://www.iec.ch/)
- [ISO 14971 Risk Management for Medical Devices](https://www.iso.org/)

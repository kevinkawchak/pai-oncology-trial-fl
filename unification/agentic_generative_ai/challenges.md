# Challenges in Unifying Agentic and Generative AI for Oncology Trials

## Overview

Agentic AI systems -- autonomous or semi-autonomous workflows composed of
language-model-driven agents -- are increasingly used in oncology clinical
trials for literature review, protocol generation, adverse event monitoring,
and treatment planning assistance. Multiple orchestration frameworks exist
(CrewAI, LangGraph, AutoGen, custom solutions), each with incompatible
tool-calling conventions, memory architectures, and execution models.

Unifying these frameworks behind a common interface is essential for
federated oncology trials where different institutions have adopted
different agentic stacks.

---

## 1. Tool-Calling Convention Fragmentation

### 1.1 Schema Incompatibility

Each framework defines tools differently:

- **CrewAI**: Tools are Python classes inheriting from `BaseTool` with
  `_run()` methods and Pydantic-based argument schemas.
- **LangGraph**: Tools are functions decorated with `@tool` or
  `StructuredTool` instances with JSON Schema argument definitions.
- **AutoGen**: Tools are registered via `register_for_llm()` with
  function signatures introspected at runtime.
- **MCP (Model Context Protocol)**: Tools use a JSON-RPC-based protocol
  with `inputSchema` conforming to JSON Schema draft-07.
- **OpenAI function calling**: Tools are JSON objects with `parameters`
  following JSON Schema.
- **Anthropic tool use**: Tools use `input_schema` with JSON Schema.

Converting between these formats without losing type information,
validation constraints, or documentation is non-trivial.

### 1.2 Async vs. Sync Execution

CrewAI tools are synchronous by default. LangGraph supports async
natively. AutoGen uses a cooperative async model. Bridging these
execution models requires careful handling of event loops, thread
pools, and cancellation semantics.

### 1.3 Tool Result Formatting

Return value conventions differ. Some frameworks expect plain strings,
others expect structured dictionaries, and MCP uses a content-block
array with MIME types. Loss of structure during conversion can degrade
agent decision quality.

---

## 2. Memory and State Management

### 2.1 Conversation History

- **CrewAI**: Agents maintain per-task memory; cross-task memory
  requires explicit configuration.
- **LangGraph**: State is managed via a typed state graph; history is
  a first-class citizen of the graph structure.
- **AutoGen**: Agents exchange messages in a group chat model;
  history is the chat transcript.

Unifying memory requires a common state representation that preserves
each framework's semantics while enabling cross-framework state
transfer (e.g., an agent started in CrewAI continuing in LangGraph
after a site migration).

### 2.2 Long-Term Knowledge

Clinical trial agents accumulate knowledge over weeks or months
(patient histories, protocol amendments, adverse event patterns).
Each framework has a different approach to persistent storage:

- Vector databases (used across frameworks but with different
  embedding models and chunking strategies)
- Graph databases (LangGraph's natural fit)
- Key-value stores (AutoGen's session management)

A unified knowledge layer must abstract over these storage backends.

### 2.3 PHI in Agent Memory

Agent memory may inadvertently capture Protected Health Information
(PHI). The unification layer must enforce de-identification at the
memory boundary, regardless of which framework manages the memory.
This is a HIPAA compliance requirement.

---

## 3. Multi-Agent Coordination

### 3.1 Orchestration Topology

- **CrewAI**: Sequential or hierarchical crew execution with a
  manager agent.
- **LangGraph**: Arbitrary directed graph with conditional edges
  and cycles.
- **AutoGen**: Group chat with speaker selection policies.

These topologies encode different assumptions about agent
collaboration. A tumor board simulation (oncologist agent +
radiologist agent + pathologist agent) has different coordination
needs than a literature screening pipeline (search agent + filter
agent + summarization agent).

### 3.2 Conflict Resolution

When multiple agents disagree (e.g., conflicting treatment
recommendations), each framework handles conflict differently:

- CrewAI relies on the manager agent or crew hierarchy.
- LangGraph encodes conflict resolution in graph edges.
- AutoGen uses speaker selection and voting.

A unified interface must define a common conflict resolution
protocol suitable for clinical decision-making.

### 3.3 Human-in-the-Loop Integration

Clinical applications require human oversight. Each framework has
different mechanisms for pausing execution, presenting options to
a clinician, and incorporating human feedback. The unification
layer must standardize this interaction pattern.

---

## 4. Model Provider Abstraction

### 4.1 LLM API Differences

Agents may use different language model providers:

- OpenAI (GPT-4o, o1, o3)
- Anthropic (Claude Opus, Sonnet)
- Google (Gemini)
- Open-source (Llama, Mistral via vLLM/Ollama)

Each provider has different:
- Token limits and pricing
- System prompt handling
- Tool-calling formats
- Streaming behavior
- Rate limiting and retry semantics

### 4.2 Model Capability Variation

Not all models support all features (vision, tool use, structured
output). The unified interface must gracefully degrade when a
model lacks a required capability, and must not assume GPT-4-level
tool use from all providers.

### 4.3 Compliance with Model Provider Terms

Some model providers prohibit medical decision-making in their
terms of service. The unification layer must track which providers
are approved for which clinical use cases.

---

## 5. Regulatory and Safety Challenges

### 5.1 Audit Trail Requirements

FDA 21 CFR Part 11 requires complete audit trails for electronic
records used in clinical trials. Every agent decision, tool
invocation, and state transition must be logged with timestamps,
user identity, and data integrity hashes.

Current frameworks have incomplete audit support:
- CrewAI logs are informal (print statements or callbacks)
- LangGraph has LangSmith tracing, which is a proprietary SaaS
- AutoGen logs are chat transcripts without structured metadata

### 5.2 Determinism and Reproducibility

LLM-based agents are inherently non-deterministic. Clinical trials
require reproducibility. The unification layer must support:
- Temperature-zero inference with seed parameters
- Cached tool results for replay
- Deterministic agent scheduling

### 5.3 Hallucination Detection

Agents operating on clinical data must not hallucinate drug names,
dosages, or patient information. Each framework has different
approaches to grounding (RAG, tool-forced responses, chain-of-thought
verification). The unified interface must provide a common
hallucination detection and mitigation layer.

### 5.4 Scope Limitation

Clinical trial agents must operate within clearly defined scope
boundaries. An adverse event monitoring agent must not make
treatment recommendations. Enforcing these boundaries consistently
across heterogeneous frameworks is challenging.

---

## 6. Performance and Scalability

### 6.1 Latency Requirements

Some oncology workflows have latency constraints:
- Intra-operative guidance: < 500 ms response
- Adverse event alerting: < 5 seconds
- Protocol generation: minutes acceptable

Different frameworks have different latency profiles. CrewAI's
sequential execution may be too slow for real-time applications.
LangGraph's parallel branches may introduce coordination overhead.

### 6.2 Cost Management

LLM API calls are expensive. A multi-agent tumor board with
five agents, each making 10 tool calls, can cost $1-5 per
invocation with GPT-4. Federated trials running thousands of
such invocations per day require cost controls that span all
backend frameworks.

### 6.3 Federation-Specific Scaling

In a federated setup, each site runs its own agent infrastructure.
Aggregating agent outputs (e.g., multi-site adverse event reports)
requires a coordination protocol that works across framework
boundaries.

---

## Mitigation Strategies

1. **Canonical tool schema**: Define a Tool dataclass that can
   serialize to MCP, OpenAI, Anthropic, CrewAI, and LangGraph
   formats. All institution-specific tools implement this schema.

2. **Agent role taxonomy**: Define an AgentRole enum with
   clinically meaningful roles (oncologist, radiologist, etc.)
   that map to framework-specific agent configurations.

3. **Audit middleware**: Insert a logging middleware at the tool
   invocation boundary that captures every call in a
   21 CFR Part 11 compliant format, regardless of framework.

4. **PHI firewall**: All tool inputs and outputs pass through a
   de-identification filter before entering agent memory.

5. **Incremental unification**: Start with CrewAI + LangGraph
   (highest adoption), then extend to AutoGen and custom backends.

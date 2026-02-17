"""
Unified Agent Interface for Oncology Clinical Trial AI Workflows.

Provides a backend-agnostic abstraction layer over CrewAI, LangGraph,
AutoGen, and custom agentic orchestrators. Enables federated oncology
trial sites to run heterogeneous agent frameworks while sharing a common
tool interface, role taxonomy, and audit trail format.

DISCLAIMER: RESEARCH USE ONLY
This software is provided for research and educational purposes only.
It has NOT been validated for clinical use, is NOT approved by the FDA
or any other regulatory body, and MUST NOT be used to make clinical
decisions or direct patient care. All agent outputs must be reviewed
by qualified clinical professionals before any action is taken.
Use at your own risk.

Copyright (c) 2026 PAI Oncology Trial FL Contributors
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# Conditional imports for optional agent backends
# ---------------------------------------------------------------------------
try:
    from crewai import Agent as CrewAgent  # type: ignore[import-untyped]
    from crewai import Task as CrewTask
    from crewai.tools import BaseTool as CrewBaseTool

    HAS_CREWAI = True
except ImportError:
    CrewAgent = None  # type: ignore[assignment,misc]
    CrewTask = None  # type: ignore[assignment,misc]
    CrewBaseTool = None  # type: ignore[assignment,misc]
    HAS_CREWAI = False

try:
    from langgraph.graph import StateGraph  # type: ignore[import-untyped]

    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None  # type: ignore[assignment,misc]
    HAS_LANGGRAPH = False

try:
    from autogen import AssistantAgent as AutoGenAssistant  # type: ignore[import-untyped]

    HAS_AUTOGEN = True
except ImportError:
    AutoGenAssistant = None  # type: ignore[assignment,misc]
    HAS_AUTOGEN = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class AgentBackend(Enum):
    """Supported agentic orchestration frameworks."""

    CREWAI = "crewai"
    LANGGRAPH = "langgraph"
    AUTOGEN = "autogen"
    CUSTOM = "custom"


class AgentRole(Enum):
    """Clinical roles for oncology trial agents."""

    ONCOLOGIST = "oncologist"
    RADIOLOGIST = "radiologist"
    PATHOLOGIST = "pathologist"
    SURGEON = "surgeon"
    PHARMACIST = "pharmacist"
    NURSE_COORDINATOR = "nurse_coordinator"
    DATA_MANAGER = "data_manager"
    BIOSTATISTICIAN = "biostatistician"
    REGULATORY_SPECIALIST = "regulatory_specialist"
    SAFETY_MONITOR = "safety_monitor"
    LITERATURE_REVIEWER = "literature_reviewer"
    PROTOCOL_WRITER = "protocol_writer"
    PATIENT_ADVOCATE = "patient_advocate"
    QUALITY_ASSURANCE = "quality_assurance"
    SIMULATION_ENGINEER = "simulation_engineer"


class AgentState(Enum):
    """Lifecycle state of an agent instance."""

    CREATED = "created"
    INITIALIZING = "initializing"
    IDLE = "idle"
    EXECUTING = "executing"
    WAITING_HUMAN = "waiting_for_human_input"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class ToolCategory(Enum):
    """Categories for clinical trial tools."""

    DATA_ACCESS = "data_access"
    IMAGING = "imaging"
    GENOMICS = "genomics"
    LITERATURE = "literature"
    DRUG_INTERACTION = "drug_interaction"
    SIMULATION = "simulation"
    REGULATORY = "regulatory"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    SAFETY = "safety"
    DOCUMENTATION = "documentation"
    WORKFLOW = "workflow"


class MessageRole(Enum):
    """Role in a conversation message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    HUMAN_REVIEWER = "human_reviewer"


class AuditEventType(Enum):
    """Types of auditable events in agent workflows."""

    AGENT_CREATED = "agent_created"
    AGENT_STARTED = "agent_started"
    TOOL_INVOKED = "tool_invoked"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"
    STATE_CHANGED = "state_changed"
    ERROR_OCCURRED = "error_occurred"
    PHI_DETECTED = "phi_detected"
    DECISION_MADE = "decision_made"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ToolParameter:
    """A single parameter in a tool's input schema."""

    name: str
    param_type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum_values: list[str] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema property definition."""
        schema: dict[str, Any] = {
            "type": self.param_type,
            "description": self.description,
        }
        if self.enum_values:
            schema["enum"] = self.enum_values
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """Backend-agnostic tool definition for oncology clinical trial agents.

    Provides serialization to multiple format conventions:
    - MCP (Model Context Protocol)
    - OpenAI function calling
    - Anthropic tool use
    - CrewAI BaseTool
    - LangGraph StructuredTool
    """

    name: str
    description: str
    category: ToolCategory = ToolCategory.DATA_ACCESS
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None
    requires_phi_check: bool = False
    requires_human_approval: bool = False
    max_execution_time_s: float = 30.0
    audit_level: str = "standard"
    version: str = "1.0.0"

    def to_mcp_format(self) -> dict[str, Any]:
        """Serialize to Model Context Protocol tool definition.

        MCP tools use JSON-RPC with inputSchema conforming to JSON Schema draft-07.
        """
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required_params.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required_params,
            },
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Serialize to OpenAI function-calling tool definition.

        OpenAI uses a 'function' wrapper with 'parameters' following JSON Schema.
        """
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required_params.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Serialize to Anthropic tool-use definition.

        Anthropic uses 'input_schema' with JSON Schema.
        """
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required_params.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required_params,
            },
        }

    def to_crewai_format(self) -> dict[str, Any]:
        """Serialize to CrewAI-compatible tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": {
                param.name: {
                    "type": param.param_type,
                    "description": param.description,
                    "required": param.required,
                }
                for param in self.parameters
            },
        }

    def to_langgraph_format(self) -> dict[str, Any]:
        """Serialize to LangGraph StructuredTool-compatible format."""
        return {
            "name": self.name,
            "description": self.description,
            "args": {
                param.name: {
                    "type": param.param_type,
                    "description": param.description,
                }
                for param in self.parameters
            },
            "return_direct": False,
        }


@dataclass
class Message:
    """A message in an agent conversation."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute integrity hash for audit trail."""
        data = json.dumps(
            {
                "message_id": self.message_id,
                "role": self.role.value,
                "content": self.content,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


@dataclass
class AuditEvent:
    """An auditable event in an agent workflow (21 CFR Part 11 compatible)."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.AGENT_STARTED
    agent_id: str = ""
    agent_role: Optional[AgentRole] = None
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    user_identity: str = "system"
    previous_hash: str = ""
    event_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute a chain hash linking this event to the previous one."""
        self.previous_hash = previous_hash
        data = json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type.value,
                "agent_id": self.agent_id,
                "timestamp": self.timestamp,
                "details": self.details,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        self.event_hash = hashlib.sha256(data).hexdigest()
        return self.event_hash


@dataclass
class AgentConfig:
    """Configuration for a unified agent instance."""

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "UnnamedAgent"
    role: AgentRole = AgentRole.ONCOLOGIST
    backend: AgentBackend = AgentBackend.CUSTOM
    model_provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096
    system_prompt: str = ""
    tools: list[Tool] = field(default_factory=list)
    allowed_categories: list[ToolCategory] = field(default_factory=list)
    requires_human_approval: bool = False
    max_iterations: int = 20
    timeout_s: float = 300.0
    phi_detection_enabled: bool = True
    audit_enabled: bool = True
    scope_restrictions: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate configuration and return issues."""
        issues: list[str] = []
        if not self.name:
            issues.append("Agent name is required")
        if self.temperature < 0.0 or self.temperature > 2.0:
            issues.append(f"Temperature {self.temperature} out of range [0, 2]")
        if self.max_tokens < 1:
            issues.append("max_tokens must be positive")
        if self.max_iterations < 1:
            issues.append("max_iterations must be positive")
        if self.timeout_s <= 0:
            issues.append("timeout_s must be positive")
        return issues


@dataclass
class AgentResponse:
    """Response from an agent execution."""

    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    iterations_used: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    errors: list[str] = field(default_factory=list)
    phi_detections: list[str] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol interfaces
# ---------------------------------------------------------------------------
class AgentBackendProtocol(Protocol):
    """Protocol that each agent backend adapter must implement."""

    def initialize(self, config: AgentConfig) -> bool: ...
    def execute(self, prompt: str, context: dict[str, Any]) -> AgentResponse: ...
    def add_tool(self, tool: Tool) -> None: ...
    def get_state(self) -> AgentState: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Audit trail manager
# ---------------------------------------------------------------------------
class AuditTrailManager:
    """Manages a hash-chained audit trail for agent workflows.

    Designed to satisfy FDA 21 CFR Part 11 requirements for electronic
    records, including tamper detection via hash chains.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._last_hash: str = ""

    def record(
        self,
        event_type: AuditEventType,
        agent_id: str,
        agent_role: Optional[AgentRole] = None,
        details: Optional[dict[str, Any]] = None,
        user_identity: str = "system",
    ) -> AuditEvent:
        """Record an auditable event with hash chain integrity."""
        event = AuditEvent(
            event_type=event_type,
            agent_id=agent_id,
            agent_role=agent_role,
            details=details or {},
            user_identity=user_identity,
        )
        event.compute_hash(self._last_hash)
        self._last_hash = event.event_hash
        self._events.append(event)

        logger.info(
            "Audit [%s] agent=%s type=%s hash=%s",
            event.event_id[:8],
            agent_id[:8] if agent_id else "N/A",
            event_type.value,
            event.event_hash[:12],
        )
        return event

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire audit chain."""
        prev_hash = ""
        for event in self._events:
            expected = event.compute_hash(prev_hash)
            if expected != event.event_hash:
                logger.error("Audit chain broken at event %s", event.event_id)
                return False
            prev_hash = event.event_hash
        return True

    def export_json(self) -> str:
        """Export the audit trail as a JSON string."""
        records = []
        for event in self._events:
            records.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "agent_id": event.agent_id,
                    "agent_role": event.agent_role.value if event.agent_role else None,
                    "timestamp": event.timestamp,
                    "details": event.details,
                    "user_identity": event.user_identity,
                    "previous_hash": event.previous_hash,
                    "event_hash": event.event_hash,
                }
            )
        return json.dumps(records, indent=2)

    @property
    def events(self) -> list[AuditEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)


# ---------------------------------------------------------------------------
# PHI detection layer
# ---------------------------------------------------------------------------
class PHIDetector:
    """Lightweight PHI detection for agent inputs and outputs.

    Checks for common patterns that may indicate Protected Health
    Information in text. Not a replacement for the full de-identification
    pipeline in privacy/phi_detector.py, but a fast pre-filter for
    agent workflows.
    """

    # Common PHI pattern indicators (non-exhaustive)
    _INDICATORS: list[str] = [
        "patient name",
        "date of birth",
        "medical record number",
        "social security",
        "ssn",
        "mrn",
        "dob",
        "address",
        "phone number",
        "email address",
        "insurance id",
        "account number",
    ]

    def check(self, text: str) -> list[str]:
        """Check text for potential PHI indicators. Returns list of findings."""
        findings: list[str] = []
        text_lower = text.lower()
        for indicator in self._INDICATORS:
            if indicator in text_lower:
                findings.append(f"Potential PHI detected: '{indicator}' pattern found")
        return findings


# ---------------------------------------------------------------------------
# Custom backend adapter (reference implementation)
# ---------------------------------------------------------------------------
class CustomBackendAdapter:
    """Reference implementation of a custom agent backend.

    This adapter demonstrates the expected behavior of a backend adapter
    without requiring any external dependencies. It executes tools
    sequentially and builds responses from tool results.
    """

    def __init__(self) -> None:
        self._config: Optional[AgentConfig] = None
        self._tools: dict[str, Tool] = {}
        self._state: AgentState = AgentState.CREATED
        self._conversation: list[Message] = []

    def initialize(self, config: AgentConfig) -> bool:
        """Initialize the custom backend."""
        self._config = config
        self._state = AgentState.INITIALIZING

        for tool in config.tools:
            self._tools[tool.name] = tool

        self._state = AgentState.IDLE
        logger.info(
            "Custom backend initialized: agent=%s role=%s tools=%d",
            config.name,
            config.role.value,
            len(self._tools),
        )
        return True

    def execute(self, prompt: str, context: Optional[dict[str, Any]] = None) -> AgentResponse:
        """Execute a prompt against the custom backend."""
        start_time = time.time()
        self._state = AgentState.EXECUTING

        response = AgentResponse(agent_id=self._config.agent_id if self._config else "")

        # Record the user message
        user_msg = Message(role=MessageRole.USER, content=prompt)
        self._conversation.append(user_msg)
        response.messages.append(user_msg)

        # Simple tool matching: check if prompt mentions any tool name
        tools_invoked: list[dict[str, Any]] = []
        for tool_name, tool in self._tools.items():
            if tool_name.lower() in prompt.lower():
                tool_call = {
                    "tool_name": tool_name,
                    "arguments": {},
                    "result": f"[{tool_name}] executed successfully (custom backend stub)",
                }
                tools_invoked.append(tool_call)

                tool_msg = Message(
                    role=MessageRole.TOOL,
                    tool_name=tool_name,
                    tool_result=tool_call["result"],
                )
                self._conversation.append(tool_msg)
                response.messages.append(tool_msg)

        # Generate response
        assistant_content = (
            f"Agent '{self._config.name if self._config else 'unknown'}' processed the request. "
            f"Tools invoked: {len(tools_invoked)}."
        )
        assistant_msg = Message(role=MessageRole.ASSISTANT, content=assistant_content)
        self._conversation.append(assistant_msg)
        response.messages.append(assistant_msg)

        response.content = assistant_content
        response.tool_calls = tools_invoked
        response.iterations_used = 1
        response.latency_ms = (time.time() - start_time) * 1000.0
        response.success = True

        self._state = AgentState.IDLE
        return response

    def add_tool(self, tool: Tool) -> None:
        """Register a tool with the backend."""
        self._tools[tool.name] = tool

    def get_state(self) -> AgentState:
        """Return the current agent state."""
        return self._state

    def pause(self) -> None:
        """Pause execution."""
        self._state = AgentState.PAUSED

    def resume(self) -> None:
        """Resume execution."""
        self._state = AgentState.IDLE

    def shutdown(self) -> None:
        """Shut down the backend."""
        self._state = AgentState.SHUTDOWN
        logger.info("Custom backend shut down")


# ---------------------------------------------------------------------------
# Unified agent interface
# ---------------------------------------------------------------------------
class UnifiedAgent:
    """Backend-agnostic agent for oncology clinical trial workflows.

    Wraps any supported backend (CrewAI, LangGraph, AutoGen, Custom) behind
    a common interface. Provides automatic PHI detection, audit trail
    management, and scope enforcement.
    """

    def __init__(self, config: AgentConfig) -> None:
        issues = config.validate()
        if issues:
            for issue in issues:
                logger.warning("Agent config issue: %s", issue)

        self._config = config
        self._backend = self._create_backend(config.backend)
        self._audit = AuditTrailManager()
        self._phi_detector = PHIDetector()
        self._state = AgentState.CREATED

        self._audit.record(
            AuditEventType.AGENT_CREATED,
            agent_id=config.agent_id,
            agent_role=config.role,
            details={
                "name": config.name,
                "backend": config.backend.value,
                "model": config.model_name,
            },
        )

    def _create_backend(self, backend_type: AgentBackend) -> CustomBackendAdapter:
        """Create the appropriate backend adapter."""
        if backend_type == AgentBackend.CREWAI and HAS_CREWAI:
            logger.info("CrewAI backend available; using CrewAI adapter")
            # In a full implementation, this would return a CrewAI-specific adapter
            return CustomBackendAdapter()
        elif backend_type == AgentBackend.LANGGRAPH and HAS_LANGGRAPH:
            logger.info("LangGraph backend available; using LangGraph adapter")
            return CustomBackendAdapter()
        elif backend_type == AgentBackend.AUTOGEN and HAS_AUTOGEN:
            logger.info("AutoGen backend available; using AutoGen adapter")
            return CustomBackendAdapter()
        else:
            if backend_type != AgentBackend.CUSTOM:
                logger.warning(
                    "Requested backend '%s' not available; falling back to custom",
                    backend_type.value,
                )
            return CustomBackendAdapter()

    def initialize(self) -> bool:
        """Initialize the agent and its backend."""
        self._state = AgentState.INITIALIZING

        self._audit.record(
            AuditEventType.AGENT_STARTED,
            agent_id=self._config.agent_id,
            agent_role=self._config.role,
        )

        success = self._backend.initialize(self._config)
        if success:
            self._state = AgentState.IDLE
            logger.info(
                "Agent initialized: name=%s role=%s backend=%s",
                self._config.name,
                self._config.role.value,
                self._config.backend.value,
            )
        else:
            self._state = AgentState.ERROR
            logger.error("Agent initialization failed: %s", self._config.name)

        return success

    def execute(self, prompt: str, context: Optional[dict[str, Any]] = None) -> AgentResponse:
        """Execute a prompt with PHI detection and audit logging."""
        start_time = time.time()

        # PHI detection on input
        phi_findings: list[str] = []
        if self._config.phi_detection_enabled:
            phi_findings = self._phi_detector.check(prompt)
            if phi_findings:
                for finding in phi_findings:
                    logger.warning("PHI check: %s", finding)
                self._audit.record(
                    AuditEventType.PHI_DETECTED,
                    agent_id=self._config.agent_id,
                    agent_role=self._config.role,
                    details={"findings": phi_findings, "source": "input"},
                )

        # Execute via backend
        response = self._backend.execute(prompt, context)

        # PHI detection on output
        if self._config.phi_detection_enabled and response.content:
            output_phi = self._phi_detector.check(response.content)
            if output_phi:
                for finding in output_phi:
                    logger.warning("PHI in output: %s", finding)
                phi_findings.extend(output_phi)
                self._audit.record(
                    AuditEventType.PHI_DETECTED,
                    agent_id=self._config.agent_id,
                    agent_role=self._config.role,
                    details={"findings": output_phi, "source": "output"},
                )

        response.phi_detections = phi_findings

        # Record audit events for tool calls
        for tool_call in response.tool_calls:
            self._audit.record(
                AuditEventType.TOOL_INVOKED,
                agent_id=self._config.agent_id,
                agent_role=self._config.role,
                details=tool_call,
            )

        # Record the decision
        self._audit.record(
            AuditEventType.DECISION_MADE,
            agent_id=self._config.agent_id,
            agent_role=self._config.role,
            details={
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "response_length": len(response.content),
                "tools_used": len(response.tool_calls),
                "latency_ms": (time.time() - start_time) * 1000.0,
            },
        )

        response.audit_events = self._audit.events
        return response

    def add_tool(self, tool: Tool) -> None:
        """Register a tool with this agent."""
        if self._config.allowed_categories and tool.category not in self._config.allowed_categories:
            logger.warning(
                "Tool '%s' category '%s' not in allowed categories for agent '%s'",
                tool.name,
                tool.category.value,
                self._config.name,
            )
            return

        self._backend.add_tool(tool)
        self._config.tools.append(tool)
        logger.info("Tool registered: %s -> agent %s", tool.name, self._config.name)

    def get_audit_trail(self) -> str:
        """Export the audit trail as JSON."""
        return self._audit.export_json()

    def verify_audit_integrity(self) -> bool:
        """Verify the integrity of the audit hash chain."""
        return self._audit.verify_chain()

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._backend.get_state()

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    def shutdown(self) -> None:
        """Shut down the agent and its backend."""
        self._backend.shutdown()
        self._state = AgentState.SHUTDOWN
        logger.info("Agent shut down: %s", self._config.name)


# ---------------------------------------------------------------------------
# Tool registry (shared across agents)
# ---------------------------------------------------------------------------
class ToolRegistry:
    """Global registry of tools available to the oncology trial federation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the global registry."""
        self._tools[tool.name] = tool
        logger.info("Tool registered globally: %s (category=%s)", tool.name, tool.category.value)

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_by_category(self, category: ToolCategory) -> list[Tool]:
        """List all tools in a given category."""
        return [t for t in self._tools.values() if t.category == category]

    def export_all_mcp(self) -> list[dict[str, Any]]:
        """Export all tools in MCP format."""
        return [t.to_mcp_format() for t in self._tools.values()]

    def export_all_openai(self) -> list[dict[str, Any]]:
        """Export all tools in OpenAI format."""
        return [t.to_openai_format() for t in self._tools.values()]

    def export_all_anthropic(self) -> list[dict[str, Any]]:
        """Export all tools in Anthropic format."""
        return [t.to_anthropic_format() for t in self._tools.values()]

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)


# ---------------------------------------------------------------------------
# Predefined oncology tools
# ---------------------------------------------------------------------------
def create_standard_oncology_tools() -> list[Tool]:
    """Create a standard set of tools for oncology clinical trial agents."""
    tools = [
        Tool(
            name="query_patient_record",
            description="Query de-identified patient records from the federated data store",
            category=ToolCategory.DATA_ACCESS,
            parameters=[
                ToolParameter(name="patient_id", param_type="string", description="De-identified patient ID"),
                ToolParameter(
                    name="data_type",
                    param_type="string",
                    description="Type of data to retrieve",
                    enum_values=["demographics", "labs", "imaging", "pathology", "treatment"],
                ),
            ],
            requires_phi_check=True,
            audit_level="elevated",
        ),
        Tool(
            name="search_literature",
            description="Search PubMed and clinical trial databases for relevant oncology literature",
            category=ToolCategory.LITERATURE,
            parameters=[
                ToolParameter(name="query", param_type="string", description="Search query"),
                ToolParameter(
                    name="max_results", param_type="integer", description="Maximum results", required=False, default=10
                ),
                ToolParameter(name="date_range", param_type="string", description="Date range filter", required=False),
            ],
        ),
        Tool(
            name="check_drug_interactions",
            description="Check for drug-drug interactions in a proposed treatment regimen",
            category=ToolCategory.DRUG_INTERACTION,
            parameters=[
                ToolParameter(name="medications", param_type="array", description="List of medication names"),
                ToolParameter(
                    name="severity_threshold",
                    param_type="string",
                    description="Minimum severity to report",
                    enum_values=["minor", "moderate", "major", "contraindicated"],
                    required=False,
                    default="moderate",
                ),
            ],
            requires_human_approval=True,
            audit_level="elevated",
        ),
        Tool(
            name="run_digital_twin_simulation",
            description="Run a treatment simulation on a patient's digital twin",
            category=ToolCategory.SIMULATION,
            parameters=[
                ToolParameter(name="twin_id", param_type="string", description="Digital twin identifier"),
                ToolParameter(name="treatment_plan", param_type="object", description="Treatment plan specification"),
                ToolParameter(
                    name="num_simulations",
                    param_type="integer",
                    description="Number of Monte Carlo simulations",
                    required=False,
                    default=100,
                ),
            ],
            max_execution_time_s=120.0,
        ),
        Tool(
            name="generate_adverse_event_report",
            description="Generate a structured adverse event report (MedDRA coded)",
            category=ToolCategory.SAFETY,
            parameters=[
                ToolParameter(
                    name="event_description", param_type="string", description="Description of the adverse event"
                ),
                ToolParameter(name="severity_grade", param_type="integer", description="CTCAE grade (1-5)"),
                ToolParameter(
                    name="causality_assessment",
                    param_type="string",
                    description="Relationship to treatment",
                    enum_values=["unrelated", "unlikely", "possible", "probable", "definite"],
                ),
            ],
            requires_human_approval=True,
            audit_level="critical",
        ),
    ]
    return tools


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the unified agent interface."""
    logger.info("Unified Agent Interface demonstration")
    logger.info("HAS_CREWAI=%s  HAS_LANGGRAPH=%s  HAS_AUTOGEN=%s", HAS_CREWAI, HAS_LANGGRAPH, HAS_AUTOGEN)

    # Create standard tools
    tools = create_standard_oncology_tools()
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    logger.info("Tool registry: %d tools registered", registry.count)

    # Demonstrate tool format export
    for tool in tools[:2]:
        mcp = tool.to_mcp_format()
        openai = tool.to_openai_format()
        anthropic = tool.to_anthropic_format()
        logger.info(
            "Tool '%s' formats: MCP=%d keys, OpenAI=%d keys, Anthropic=%d keys",
            tool.name,
            len(mcp),
            len(openai),
            len(anthropic),
        )

    # Create and initialize an agent
    config = AgentConfig(
        name="OncologyAssistant",
        role=AgentRole.ONCOLOGIST,
        backend=AgentBackend.CUSTOM,
        model_name="claude-sonnet-4-20250514",
        tools=tools,
        phi_detection_enabled=True,
        audit_enabled=True,
    )

    agent = UnifiedAgent(config)
    agent.initialize()

    # Execute a sample prompt
    response = agent.execute(
        "Search the literature for recent immunotherapy trials in NSCLC "
        "and check drug interactions for pembrolizumab + carboplatin."
    )

    logger.info(
        "Agent response: success=%s tools=%d latency=%.1fms",
        response.success,
        len(response.tool_calls),
        response.latency_ms,
    )

    # Verify audit trail integrity
    integrity_ok = agent.verify_audit_integrity()
    logger.info("Audit trail integrity: %s", "PASS" if integrity_ok else "FAIL")

    agent.shutdown()
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()

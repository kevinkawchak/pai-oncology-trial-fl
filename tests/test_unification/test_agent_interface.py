"""Tests for unification/agentic_generative_ai/unified_agent_interface.py.

Covers enums (AgentBackend, AgentRole, AgentState, ToolCategory, MessageRole,
AuditEventType), dataclasses (ToolParameter, Tool, Message, AuditEvent,
AgentConfig, AgentResponse), AuditTrailManager, PHIDetector,
CustomBackendAdapter, UnifiedAgent, ToolRegistry, and factory functions.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import load_module

mod = load_module(
    "unified_agent_interface",
    "unification/agentic_generative_ai/unified_agent_interface.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestAgentBackend:
    """Tests for the AgentBackend enum."""

    def test_members(self):
        """AgentBackend has four members."""
        expected = {"CREWAI", "LANGGRAPH", "AUTOGEN", "CUSTOM"}
        assert set(mod.AgentBackend.__members__.keys()) == expected


class TestAgentRole:
    """Tests for the AgentRole enum."""

    def test_count(self):
        """AgentRole has 15 members."""
        assert len(mod.AgentRole) == 15

    def test_common_roles(self):
        """Key clinical roles exist."""
        assert mod.AgentRole.ONCOLOGIST.value == "oncologist"
        assert mod.AgentRole.RADIOLOGIST.value == "radiologist"
        assert mod.AgentRole.SAFETY_MONITOR.value == "safety_monitor"


class TestAgentState:
    """Tests for the AgentState enum."""

    def test_members(self):
        """AgentState has nine members."""
        expected = {
            "CREATED",
            "INITIALIZING",
            "IDLE",
            "EXECUTING",
            "WAITING_HUMAN",
            "PAUSED",
            "COMPLETED",
            "ERROR",
            "SHUTDOWN",
        }
        assert set(mod.AgentState.__members__.keys()) == expected

    def test_initial_state(self):
        """CREATED is the initial state."""
        assert mod.AgentState.CREATED.value == "created"


class TestToolCategory:
    """Tests for the ToolCategory enum."""

    def test_count(self):
        """ToolCategory has 12 members."""
        assert len(mod.ToolCategory) == 12

    def test_common_categories(self):
        """Key tool categories exist."""
        assert mod.ToolCategory.DATA_ACCESS.value == "data_access"
        assert mod.ToolCategory.SAFETY.value == "safety"
        assert mod.ToolCategory.IMAGING.value == "imaging"


class TestMessageRole:
    """Tests for the MessageRole enum."""

    def test_members(self):
        """MessageRole has five members."""
        expected = {"SYSTEM", "USER", "ASSISTANT", "TOOL", "HUMAN_REVIEWER"}
        assert set(mod.MessageRole.__members__.keys()) == expected


class TestAuditEventType:
    """Tests for the AuditEventType enum."""

    def test_count(self):
        """AuditEventType has 13 members."""
        assert len(mod.AuditEventType) == 13

    def test_key_types(self):
        """Key event types exist."""
        assert mod.AuditEventType.AGENT_CREATED.value == "agent_created"
        assert mod.AuditEventType.TOOL_INVOKED.value == "tool_invoked"
        assert mod.AuditEventType.PHI_DETECTED.value == "phi_detected"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestToolParameter:
    """Tests for the ToolParameter dataclass."""

    def test_defaults(self):
        """Default ToolParameter is a required string."""
        tp = mod.ToolParameter(name="dose", param_type="number", description="Dose in Gy")
        assert tp.name == "dose"
        assert tp.required is True

    def test_optional(self):
        """ToolParameter can be optional."""
        tp = mod.ToolParameter(name="notes", param_type="string", description="Notes", required=False)
        assert tp.required is False

    def test_to_json_schema(self):
        """to_json_schema produces a dict with type and description."""
        tp = mod.ToolParameter(name="x", param_type="number", description="A number")
        schema = tp.to_json_schema()
        assert schema["type"] == "number"
        assert schema["description"] == "A number"


class TestTool:
    """Tests for the Tool dataclass."""

    def test_to_mcp_format(self):
        """to_mcp_format produces a dict with 'name' and 'inputSchema'."""
        t = mod.Tool(
            name="compute_dose",
            description="Compute radiation dose",
            category=mod.ToolCategory.DATA_ACCESS,
            parameters=[mod.ToolParameter(name="dose", param_type="number", description="dose")],
        )
        mcp = t.to_mcp_format()
        assert mcp["name"] == "compute_dose"
        assert "inputSchema" in mcp

    def test_to_openai_format(self):
        """to_openai_format wraps tool as an OpenAI function tool."""
        t = mod.Tool(
            name="check",
            description="Run check",
            category=mod.ToolCategory.SAFETY,
            parameters=[],
        )
        openai = t.to_openai_format()
        assert openai["type"] == "function"
        assert openai["function"]["name"] == "check"

    def test_to_anthropic_format(self):
        """to_anthropic_format returns dict with 'name' and 'input_schema'."""
        t = mod.Tool(
            name="audit",
            description="Audit action",
            category=mod.ToolCategory.ANALYTICS,
            parameters=[],
        )
        anthropic = t.to_anthropic_format()
        assert anthropic["name"] == "audit"
        assert "input_schema" in anthropic

    def test_to_crewai_format(self):
        """to_crewai_format returns dict with 'name' key."""
        t = mod.Tool(name="t1", description="d", category=mod.ToolCategory.DATA_ACCESS, parameters=[])
        crewai = t.to_crewai_format()
        assert crewai["name"] == "t1"

    def test_to_langgraph_format(self):
        """to_langgraph_format returns dict with 'name' and 'args' keys."""
        t = mod.Tool(name="t2", description="d", category=mod.ToolCategory.DATA_ACCESS, parameters=[])
        lg = t.to_langgraph_format()
        assert lg["name"] == "t2"
        assert "args" in lg


class TestAgentConfig:
    """Tests for the AgentConfig dataclass."""

    def test_defaults(self):
        """Default config has valid values."""
        cfg = mod.AgentConfig(
            name="test_agent",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        issues = cfg.validate()
        assert len(issues) == 0

    def test_empty_name_invalid(self):
        """Empty name produces a validation issue."""
        cfg = mod.AgentConfig(
            name="",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        issues = cfg.validate()
        assert any("name" in i.lower() for i in issues)

    def test_max_tokens_clamped(self):
        """Negative max_tokens produces validation issue."""
        cfg = mod.AgentConfig(
            name="a",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
            max_tokens=-1,
        )
        issues = cfg.validate()
        assert any("max_tokens" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# AuditTrailManager
# ---------------------------------------------------------------------------
class TestAuditTrailManager:
    """Tests for the AuditTrailManager class."""

    def test_record_event(self):
        """record adds an event to the trail."""
        atm = mod.AuditTrailManager()
        atm.record(mod.AuditEventType.AGENT_CREATED, agent_id="a1", details={"key": "val"})
        assert len(atm.events) == 1
        assert atm.events[0].event_type == mod.AuditEventType.AGENT_CREATED

    def test_chain_integrity(self):
        """verify_chain returns True for sequential events."""
        atm = mod.AuditTrailManager()
        atm.record(mod.AuditEventType.AGENT_CREATED, agent_id="a2")
        atm.record(mod.AuditEventType.TOOL_INVOKED, agent_id="a2")
        assert atm.verify_chain() is True

    def test_export_json(self):
        """export_json returns valid JSON."""
        atm = mod.AuditTrailManager()
        atm.record(mod.AuditEventType.AGENT_CREATED, agent_id="a3")
        exported = atm.export_json()
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_empty_trail_verifies(self):
        """Empty audit trail verifies as True."""
        atm = mod.AuditTrailManager()
        assert atm.verify_chain() is True


# ---------------------------------------------------------------------------
# PHIDetector
# ---------------------------------------------------------------------------
class TestPHIDetector:
    """Tests for the PHIDetector class."""

    def test_no_phi(self):
        """Clean text has no PHI."""
        detector = mod.PHIDetector()
        findings = detector.check("Patient presented with grade 2 toxicity.")
        assert len(findings) == 0

    def test_ssn_detected(self):
        """SSN keyword is detected."""
        detector = mod.PHIDetector()
        findings = detector.check("SSN: 123-45-6789")
        assert len(findings) > 0
        assert any("ssn" in f.lower() or "social" in f.lower() for f in findings)

    def test_phone_number_keyword_detected(self):
        """Phone number keyword is detected."""
        detector = mod.PHIDetector()
        findings = detector.check("phone number is 555-123-4567")
        assert len(findings) > 0

    def test_email_keyword_detected(self):
        """Email address keyword is detected."""
        detector = mod.PHIDetector()
        findings = detector.check("email address: john@hospital.org")
        assert len(findings) > 0

    def test_mrn_detected(self):
        """MRN keyword is detected."""
        detector = mod.PHIDetector()
        findings = detector.check("MRN 12345678")
        assert len(findings) > 0

    def test_dob_detected(self):
        """DOB keyword is detected."""
        detector = mod.PHIDetector()
        findings = detector.check("DOB: 1990-01-01")
        assert len(findings) > 0


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------
class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_and_get(self):
        """Register then get returns the tool."""
        registry = mod.ToolRegistry()
        tool = mod.Tool(
            name="mytool",
            description="test tool",
            category=mod.ToolCategory.DATA_ACCESS,
            parameters=[],
        )
        registry.register(tool)
        assert registry.get("mytool") is not None
        assert registry.get("mytool").name == "mytool"

    def test_get_missing_returns_none(self):
        """Get non-existent tool returns None."""
        registry = mod.ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_by_category(self):
        """list_by_category filters correctly."""
        registry = mod.ToolRegistry()
        t1 = mod.Tool(name="a", description="d", category=mod.ToolCategory.DATA_ACCESS, parameters=[])
        t2 = mod.Tool(name="b", description="d", category=mod.ToolCategory.SAFETY, parameters=[])
        registry.register(t1)
        registry.register(t2)
        result = registry.list_by_category(mod.ToolCategory.DATA_ACCESS)
        assert len(result) == 1
        assert result[0].name == "a"

    def test_count(self):
        """count returns the number of registered tools."""
        registry = mod.ToolRegistry()
        assert registry.count == 0
        t = mod.Tool(name="x", description="d", category=mod.ToolCategory.ANALYTICS, parameters=[])
        registry.register(t)
        assert registry.count == 1

    def test_export_all_mcp(self):
        """export_all_mcp returns list of MCP format dicts."""
        registry = mod.ToolRegistry()
        t = mod.Tool(name="z", description="d", category=mod.ToolCategory.IMAGING, parameters=[])
        registry.register(t)
        mcp_list = registry.export_all_mcp()
        assert len(mcp_list) == 1
        assert mcp_list[0]["name"] == "z"


# ---------------------------------------------------------------------------
# UnifiedAgent
# ---------------------------------------------------------------------------
class TestUnifiedAgent:
    """Tests for the UnifiedAgent class."""

    def test_creation(self):
        """UnifiedAgent starts in CREATED state."""
        cfg = mod.AgentConfig(
            name="agent1",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        agent = mod.UnifiedAgent(cfg)
        assert agent.state == mod.AgentState.CREATED

    def test_initialize(self):
        """Initialize transitions to IDLE."""
        cfg = mod.AgentConfig(
            name="agent2",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        agent = mod.UnifiedAgent(cfg)
        agent.initialize()
        assert agent.state == mod.AgentState.IDLE

    def test_shutdown(self):
        """Shutdown transitions to SHUTDOWN."""
        cfg = mod.AgentConfig(
            name="agent3",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        agent = mod.UnifiedAgent(cfg)
        agent.initialize()
        agent.shutdown()
        assert agent.state == mod.AgentState.SHUTDOWN

    def test_add_tool(self):
        """Tools can be added to the agent."""
        cfg = mod.AgentConfig(
            name="agent4",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        agent = mod.UnifiedAgent(cfg)
        agent.initialize()
        tool = mod.Tool(name="calc", description="d", category=mod.ToolCategory.IMAGING, parameters=[])
        agent.add_tool(tool)
        # No exception means success

    def test_config_property(self):
        """config property returns the original AgentConfig."""
        cfg = mod.AgentConfig(
            name="agent5",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        agent = mod.UnifiedAgent(cfg)
        assert agent.config.name == "agent5"

    def test_audit_trail_after_init(self):
        """Audit trail records initialization event."""
        cfg = mod.AgentConfig(
            name="agent6",
            role=mod.AgentRole.ONCOLOGIST,
            backend=mod.AgentBackend.CUSTOM,
        )
        agent = mod.UnifiedAgent(cfg)
        agent.initialize()
        trail = agent.get_audit_trail()
        assert len(trail) > 0


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
class TestCreateStandardOncologyTools:
    """Tests for create_standard_oncology_tools factory."""

    def test_returns_list(self):
        """Returns a non-empty list of tools."""
        tools = mod.create_standard_oncology_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tools_have_names(self):
        """Each tool has a non-empty name."""
        for tool in mod.create_standard_oncology_tools():
            assert tool.name != ""

    def test_tools_have_categories(self):
        """Each tool has a valid category."""
        for tool in mod.create_standard_oncology_tools():
            assert isinstance(tool.category, mod.ToolCategory)

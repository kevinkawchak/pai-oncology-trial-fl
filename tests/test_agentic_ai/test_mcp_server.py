"""Tests for agentic-ai/examples-agentic-ai/01_mcp_oncology_server.py.

Validates MCP server configuration, tool registration, resource definitions,
audit trail integrity, tool invocation handlers, and dataclass serialization.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "mcp_oncology_server",
    "agentic-ai/examples-agentic-ai/01_mcp_oncology_server.py",
)


class TestEnums:
    """Verify all enum types are defined with expected members."""

    def test_robot_mode_values(self):
        assert mod.RobotMode.IDLE.value == "idle"
        assert mod.RobotMode.OPERATING.value == "operating"
        assert mod.RobotMode.EMERGENCY_STOP.value == "emergency_stop"

    def test_procedure_phase_values(self):
        assert mod.ProcedurePhase.PRE_OPERATIVE.value == "pre_operative"
        assert mod.ProcedurePhase.RESECTION.value == "resection"
        assert mod.ProcedurePhase.POST_OPERATIVE.value == "post_operative"

    def test_compliance_status_values(self):
        assert mod.ComplianceStatus.COMPLIANT.value == "compliant"
        assert mod.ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert mod.ComplianceStatus.NEEDS_REVIEW.value == "needs_review"

    def test_tool_category_values(self):
        assert mod.ToolCategory.PATIENT_DATA.value == "patient_data"
        assert mod.ToolCategory.TREATMENT_SIMULATION.value == "treatment_simulation"
        assert mod.ToolCategory.SAFETY.value == "safety"

    def test_audit_event_type_values(self):
        assert mod.AuditEventType.TOOL_INVOKED.value == "tool_invoked"
        assert mod.AuditEventType.TOOL_COMPLETED.value == "tool_completed"
        assert mod.AuditEventType.SERVER_STARTED.value == "server_started"


class TestDataclasses:
    """Verify dataclass creation and serialization."""

    def test_robot_telemetry_defaults(self):
        tel = mod.RobotTelemetry()
        assert tel.mode == mod.RobotMode.IDLE
        assert tel.phase == mod.ProcedurePhase.PRE_OPERATIVE
        assert len(tel.position_mm) == 3
        assert tel.battery_pct == 100.0

    def test_robot_telemetry_is_safe_default(self):
        tel = mod.RobotTelemetry()
        assert tel.is_safe() is True

    def test_robot_telemetry_unsafe_on_estop(self):
        tel = mod.RobotTelemetry()
        tel.safety_flags["e_stop_active"] = True
        assert tel.is_safe() is False

    def test_robot_telemetry_to_dict(self):
        tel = mod.RobotTelemetry(robot_id="R1")
        d = tel.to_dict()
        assert d["robot_id"] == "R1"
        assert "is_safe" in d

    def test_patient_record_to_dict(self):
        pr = mod.PatientRecord(patient_id="PAT-001", diagnosis="NSCLC", stage="IIIB")
        d = pr.to_dict()
        assert d["patient_id"] == "PAT-001"
        assert d["diagnosis"] == "NSCLC"

    def test_trial_protocol_to_dict(self):
        proto = mod.TrialProtocol(protocol_id="P1", title="Test Trial", phase="II")
        d = proto.to_dict()
        assert d["protocol_id"] == "P1"
        assert d["phase"] == "II"

    def test_compliance_result_to_dict(self):
        cr = mod.ComplianceResult(status=mod.ComplianceStatus.COMPLIANT, protocol_id="P1")
        d = cr.to_dict()
        assert d["status"] == "compliant"

    def test_simulation_result_to_dict(self):
        sr = mod.SimulationResult(patient_id="PAT-001", modality="chemotherapy")
        d = sr.to_dict()
        assert d["patient_id"] == "PAT-001"
        assert d["modality"] == "chemotherapy"


class TestAuditTrailManager:
    """Verify audit trail hash chaining and integrity verification."""

    def test_audit_record_creates_entry(self):
        mgr = mod.AuditTrailManager()
        entry = mgr.record(mod.AuditEventType.TOOL_INVOKED, tool_name="test_tool")
        assert entry.entry_hash != ""
        assert mgr.count == 1

    def test_audit_chain_integrity(self):
        mgr = mod.AuditTrailManager()
        mgr.record(mod.AuditEventType.SERVER_STARTED, tool_name="server")
        mgr.record(mod.AuditEventType.TOOL_INVOKED, tool_name="lookup")
        mgr.record(mod.AuditEventType.TOOL_COMPLETED, tool_name="lookup")
        assert mgr.verify_chain() is True

    def test_audit_export_json(self):
        mgr = mod.AuditTrailManager()
        mgr.record(mod.AuditEventType.TOOL_INVOKED, tool_name="t1")
        exported = mgr.export_json()
        parsed = json.loads(exported)
        assert len(parsed) == 1
        assert parsed[0]["tool_name"] == "t1"


class TestOncologyMCPServer:
    """Verify the full MCP server: tools, resources, invocations, audit."""

    def test_server_initialization(self):
        server = mod.OncologyMCPServer(server_name="test-server")
        assert server.tool_count == 4
        assert server.resource_count == 4
        assert server.audit_entry_count >= 1  # SERVER_STARTED recorded

    def test_list_tools_returns_definitions(self):
        server = mod.OncologyMCPServer()
        tools = server.list_tools()
        assert len(tools) == 4
        names = [t["name"] for t in tools]
        assert "oncology_patient_lookup" in names
        assert "oncology_treatment_simulation" in names

    def test_list_resources_returns_definitions(self):
        server = mod.OncologyMCPServer()
        resources = server.list_resources()
        assert len(resources) == 4
        uris = [r["uri"] for r in resources]
        assert any("protocols" in u for u in uris)

    def test_call_tool_patient_lookup(self):
        server = mod.OncologyMCPServer(seed=42)
        result = server.call_tool("oncology_patient_lookup", {"diagnosis": "NSCLC"})
        assert result["status"] in ("found", "no_matches")
        assert "count" in result

    def test_call_tool_treatment_simulation(self):
        server = mod.OncologyMCPServer(seed=42)
        result = server.call_tool(
            "oncology_treatment_simulation",
            {"patient_id": "FAKE-PAT", "modality": "chemotherapy", "cycles": 2},
        )
        assert "predicted_volume_cm3" in result
        assert "response_category" in result

    def test_call_tool_unknown_returns_error(self):
        server = mod.OncologyMCPServer()
        result = server.call_tool("nonexistent_tool")
        assert "error" in result

    def test_read_resource_protocol(self):
        server = mod.OncologyMCPServer()
        result = server.read_resource("oncology://protocols/PROTO-ONC-2025-001")
        assert "content" in result
        content = json.loads(result["content"])
        assert "protocol_id" in content

    def test_read_resource_not_found(self):
        server = mod.OncologyMCPServer()
        result = server.read_resource("oncology://nonexistent/resource")
        assert "error" in result

    def test_verify_audit_integrity_after_operations(self):
        server = mod.OncologyMCPServer(seed=42)
        server.call_tool("oncology_patient_lookup", {})
        server.call_tool("oncology_robot_telemetry", {})
        assert server.verify_audit_integrity() is True

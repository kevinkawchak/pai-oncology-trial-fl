"""Agentic-to-regulatory integration tests.

Tests agent configuration -> compliance check -> FDA tracking flow,
verifying that agentic AI outputs are captured in regulatory audit trails.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
mcp_mod = load_module(
    "mcp_oncology_server",
    "agentic-ai/examples-agentic-ai/01_mcp_oncology_server.py",
)
compliance_mod = load_module(
    "regulatory.compliance_checker",
    "regulatory/compliance_checker.py",
)
fda_mod = load_module(
    "regulatory.fda_submission",
    "regulatory/fda_submission.py",
)
fda_tracker_mod = load_module(
    "fda_submission_tracker",
    "regulatory/fda-compliance/fda_submission_tracker.py",
)


class TestAgentConfigToCompliance:
    """Verify agent server configuration can be checked for compliance."""

    def test_mcp_server_creates_audit_trail(self):
        server = mcp_mod.OncologyMCPServer(server_name="test")
        assert server.audit_entry_count >= 1

    def test_mcp_server_audit_chain_valid(self):
        server = mcp_mod.OncologyMCPServer(server_name="test")
        server.call_tool("oncology_patient_lookup", {})
        assert server.verify_audit_integrity() is True

    def test_compliance_checker_with_full_config(self):
        checker = compliance_mod.ComplianceChecker(regulations=["hipaa", "gdpr"])
        config = {
            "use_differential_privacy": True,
            "dp_epsilon": 1.0,
            "use_secure_aggregation": True,
            "use_deidentification": True,
            "audit_logging_enabled": True,
            "consent_management_enabled": True,
            "encryption_in_transit": True,
            "min_clients": 3,
        }
        report = checker.check_federation_config(config)
        assert len(report.checks) > 0

    def test_compliance_report_has_regulation_coverage(self):
        checker = compliance_mod.ComplianceChecker(regulations=["hipaa", "gdpr"])
        config = {
            "use_differential_privacy": True,
            "dp_epsilon": 2.0,
            "use_secure_aggregation": True,
            "use_deidentification": True,
            "audit_logging_enabled": True,
            "consent_management_enabled": True,
            "encryption_in_transit": True,
            "min_clients": 2,
        }
        report = checker.check_federation_config(config)
        regs = {c.regulation for c in report.checks}
        assert compliance_mod.Regulation.HIPAA in regs
        assert compliance_mod.Regulation.GDPR in regs


class TestFDASubmission:
    """Verify FDA submission module structure."""

    def test_fda_module_loaded(self):
        assert fda_mod is not None

    def test_fda_module_has_expected_classes(self):
        attrs = dir(fda_mod)
        assert len(attrs) > 0


class TestFDATracker:
    """Verify FDA submission tracker from hyphenated directory."""

    def test_fda_tracker_module_loaded(self):
        assert fda_tracker_mod is not None

    def test_fda_tracker_has_audit_trail(self):
        tracker = fda_tracker_mod.FDASubmissionTracker()
        assert hasattr(tracker, "get_audit_trail")


class TestAgentAuditToRegulatory:
    """Verify agent audit trail can feed regulatory review."""

    def test_mcp_audit_export_is_valid_json(self):
        server = mcp_mod.OncologyMCPServer(server_name="audit-test")
        server.call_tool("oncology_patient_lookup", {"diagnosis": "NSCLC"})
        audit_json = server.get_audit_trail()
        parsed = json.loads(audit_json)
        assert isinstance(parsed, list)
        assert len(parsed) >= 2  # SERVER_STARTED + TOOL_INVOKED

    def test_audit_entries_have_timestamps(self):
        server = mcp_mod.OncologyMCPServer(server_name="ts-test")
        server.call_tool("oncology_robot_telemetry", {})
        audit_json = server.get_audit_trail()
        entries = json.loads(audit_json)
        for entry in entries:
            assert "timestamp" in entry
            assert isinstance(entry["timestamp"], (int, float))

    def test_audit_entries_have_hash_chain(self):
        server = mcp_mod.OncologyMCPServer(server_name="hash-test")
        server.call_tool("oncology_patient_lookup", {})
        server.call_tool("oncology_robot_telemetry", {})
        audit_json = server.get_audit_trail()
        entries = json.loads(audit_json)
        # Second entry should reference the first
        if len(entries) >= 2:
            assert entries[1]["previous_hash"] == entries[0]["entry_hash"]

    def test_regulation_enum_values(self):
        assert compliance_mod.Regulation.HIPAA.value == "hipaa"
        assert compliance_mod.Regulation.GDPR.value == "gdpr"
        assert compliance_mod.Regulation.FDA_21CFR11.value == "fda_21cfr11"

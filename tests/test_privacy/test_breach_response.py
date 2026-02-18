"""Tests for privacy/breach_response.py — Breach detection and response.

Covers BreachResponseProtocol creation, indicator reporting,
auto-escalation, incident lifecycle (detect -> investigate ->
contain -> resolve -> report), severity assessment, and query methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("privacy.breach_response", "privacy/breach_response.py")

BreachResponseProtocol = mod.BreachResponseProtocol
BreachSeverity = mod.BreachSeverity
IncidentStatus = mod.IncidentStatus
BreachIndicator = mod.BreachIndicator
BreachIncident = mod.BreachIncident


class TestBreachSeverityEnum:
    """Tests for BreachSeverity enum."""

    def test_severity_values(self):
        """All four severity levels exist."""
        assert BreachSeverity.LOW.value == "low"
        assert BreachSeverity.MEDIUM.value == "medium"
        assert BreachSeverity.HIGH.value == "high"
        assert BreachSeverity.CRITICAL.value == "critical"


class TestIncidentStatusEnum:
    """Tests for IncidentStatus lifecycle enum."""

    def test_status_values(self):
        """All five lifecycle statuses exist."""
        assert IncidentStatus.DETECTED.value == "detected"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.CONTAINED.value == "contained"
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert IncidentStatus.REPORTED.value == "reported"


class TestBreachResponseProtocolCreation:
    """Tests for BreachResponseProtocol initialization."""

    def test_default_creation(self):
        """Default protocol has threshold=3 and window=60."""
        protocol = BreachResponseProtocol()
        assert protocol.auto_escalate_threshold == 3
        assert protocol.escalation_window_minutes == 60

    def test_custom_threshold(self):
        """Custom escalation threshold is accepted."""
        protocol = BreachResponseProtocol(auto_escalate_threshold=5)
        assert protocol.auto_escalate_threshold == 5

    def test_no_initial_incidents(self):
        """Newly created protocol has no open incidents."""
        protocol = BreachResponseProtocol()
        assert protocol.get_open_incidents() == []

    def test_no_initial_indicators(self):
        """Newly created protocol has no indicators."""
        protocol = BreachResponseProtocol()
        assert protocol.get_all_indicators() == []


class TestIndicatorReporting:
    """Tests for reporting breach indicators."""

    def test_report_single_indicator(self):
        """Reporting one indicator does not auto-escalate (threshold=3)."""
        protocol = BreachResponseProtocol()
        indicator = BreachIndicator(
            indicator_type="anomalous_access",
            description="Unusual data access pattern",
            severity=BreachSeverity.LOW,
        )
        result = protocol.report_indicator(indicator)
        assert result is None
        assert len(protocol.get_all_indicators()) == 1

    def test_auto_escalation_at_threshold(self):
        """Auto-escalation triggers when indicators reach threshold."""
        protocol = BreachResponseProtocol(auto_escalate_threshold=3)
        for i in range(3):
            indicator = BreachIndicator(
                indicator_type="failed_auth",
                description=f"Failed auth attempt {i + 1}",
                severity=BreachSeverity.MEDIUM,
            )
            result = protocol.report_indicator(indicator)

        # Third indicator should trigger auto-escalation
        assert result is not None
        assert result.startswith("INC-")

    def test_auto_escalation_picks_highest_severity(self):
        """Auto-escalated incident gets the highest severity from indicators."""
        protocol = BreachResponseProtocol(auto_escalate_threshold=3)
        severities = [BreachSeverity.LOW, BreachSeverity.CRITICAL, BreachSeverity.MEDIUM]
        result = None
        for i, sev in enumerate(severities):
            indicator = BreachIndicator(
                indicator_type="test",
                description=f"Indicator {i + 1}",
                severity=sev,
            )
            result = protocol.report_indicator(indicator)

        assert result is not None
        incident = protocol.get_incident(result)
        assert incident is not None
        assert incident.severity == BreachSeverity.CRITICAL


class TestIncidentLifecycle:
    """Tests for incident lifecycle transitions."""

    def test_create_incident(self):
        """Manually created incident has DETECTED status."""
        protocol = BreachResponseProtocol()
        iid = protocol.create_incident(severity=BreachSeverity.HIGH)
        assert iid.startswith("INC-")
        incident = protocol.get_incident(iid)
        assert incident is not None
        assert incident.status == IncidentStatus.DETECTED

    def test_investigate_incident(self):
        """Investigating moves incident to INVESTIGATING status."""
        protocol = BreachResponseProtocol()
        iid = protocol.create_incident()
        assert protocol.investigate(iid, notes="Starting investigation") is True
        incident = protocol.get_incident(iid)
        assert incident.status == IncidentStatus.INVESTIGATING
        assert len(incident.response_actions) == 1

    def test_contain_incident(self):
        """Containing sets status and contained_at timestamp."""
        protocol = BreachResponseProtocol()
        iid = protocol.create_incident()
        protocol.investigate(iid)
        assert protocol.contain(iid, actions_taken="Isolated affected systems") is True
        incident = protocol.get_incident(iid)
        assert incident.status == IncidentStatus.CONTAINED
        assert incident.contained_at is not None

    def test_resolve_incident(self):
        """Resolving sets status, resolved_at, and root_cause."""
        protocol = BreachResponseProtocol()
        iid = protocol.create_incident()
        protocol.investigate(iid)
        protocol.contain(iid)
        assert protocol.resolve(iid, root_cause="Misconfigured firewall") is True
        incident = protocol.get_incident(iid)
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None
        assert incident.root_cause == "Misconfigured firewall"

    def test_report_to_authorities(self):
        """Reporting sets status to REPORTED with timestamp."""
        protocol = BreachResponseProtocol()
        iid = protocol.create_incident()
        assert protocol.report_to_authorities(iid, authority="HHS_OCR") is True
        incident = protocol.get_incident(iid)
        assert incident.status == IncidentStatus.REPORTED
        assert incident.reported_at is not None

    def test_lifecycle_on_unknown_incident(self):
        """Operations on unknown incident IDs return False."""
        protocol = BreachResponseProtocol()
        assert protocol.investigate("INC-9999") is False
        assert protocol.contain("INC-9999") is False
        assert protocol.resolve("INC-9999") is False
        assert protocol.report_to_authorities("INC-9999") is False


class TestIncidentQueries:
    """Tests for querying incidents."""

    def test_get_open_incidents(self):
        """Open incidents exclude RESOLVED and REPORTED."""
        protocol = BreachResponseProtocol()
        iid1 = protocol.create_incident()
        iid2 = protocol.create_incident()
        protocol.resolve(iid1, root_cause="fixed")
        open_inc = protocol.get_open_incidents()
        assert len(open_inc) == 1
        assert open_inc[0].incident_id == iid2

    def test_get_incident_none_for_missing(self):
        """get_incident returns None for non-existent ID."""
        protocol = BreachResponseProtocol()
        assert protocol.get_incident("INC-FAKE") is None

    def test_incident_has_detected_at(self):
        """Created incidents have a non-empty detected_at timestamp."""
        protocol = BreachResponseProtocol()
        iid = protocol.create_incident(affected_sites=["SITE-A", "SITE-B"])
        incident = protocol.get_incident(iid)
        assert incident.detected_at != ""
        assert incident.affected_sites == ["SITE-A", "SITE-B"]

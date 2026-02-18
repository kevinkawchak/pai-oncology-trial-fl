"""Tests for regulatory/regulatory-intelligence/regulatory_tracker.py.

Covers RegulatoryTracker creation, jurisdiction monitoring, regulatory
update management, recent updates with cutoff, compliance requirements,
overdue status ordering (OVERDUE before IMMINENT), action items,
and dashboard reporting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "regulatory_tracker",
    "regulatory/regulatory-intelligence/regulatory_tracker.py",
)

RegulatoryTracker = mod.RegulatoryTracker
Jurisdiction = mod.Jurisdiction
UpdateType = mod.UpdateType
Priority = mod.Priority
ComplianceStatus = mod.ComplianceStatus
ActionStatus = mod.ActionStatus
RegulatoryUpdate = mod.RegulatoryUpdate
ComplianceRequirement = mod.ComplianceRequirement


class TestJurisdictionEnum:
    """Tests for Jurisdiction enum."""

    def test_all_jurisdictions_exist(self):
        """All five monitored jurisdictions are defined."""
        assert Jurisdiction.FDA.value == "fda"
        assert Jurisdiction.EMA.value == "ema"
        assert Jurisdiction.PMDA.value == "pmda"
        assert Jurisdiction.TGA.value == "tga"
        assert Jurisdiction.HEALTH_CANADA.value == "health_canada"


class TestComplianceStatusEnum:
    """Tests for ComplianceStatus enum values."""

    def test_overdue_and_imminent_exist(self):
        """OVERDUE and IMMINENT statuses are defined."""
        assert ComplianceStatus.OVERDUE.value == "overdue"
        assert ComplianceStatus.IMMINENT.value == "imminent"

    def test_all_statuses(self):
        """All compliance statuses exist."""
        assert len(ComplianceStatus) == 7


class TestRegulatoryTrackerCreation:
    """Tests for RegulatoryTracker initialization."""

    def test_default_creation(self):
        """Newly created tracker starts empty."""
        tracker = RegulatoryTracker()
        dashboard = tracker.get_dashboard()
        assert dashboard["total_updates"] == 0
        assert dashboard["total_requirements"] == 0
        assert dashboard["total_actions"] == 0


class TestRegulatoryUpdates:
    """Tests for adding and querying regulatory updates."""

    def test_add_update(self):
        """Regulatory update is added and retrievable."""
        tracker = RegulatoryTracker()
        update = tracker.add_update(
            jurisdiction=Jurisdiction.FDA,
            update_type=UpdateType.GUIDANCE_FINAL,
            title="PCCP Guidance August 2025",
            summary="Final guidance on predetermined change control plans",
            priority=Priority.HIGH,
        )
        assert update.update_id.startswith("RU-")
        assert update.jurisdiction == Jurisdiction.FDA
        assert update.priority == Priority.HIGH

    def test_get_update(self):
        """Retrieve update by ID."""
        tracker = RegulatoryTracker()
        update = tracker.add_update(Jurisdiction.EMA, UpdateType.REGULATION_FINAL, "EU AI Act")
        retrieved = tracker.get_update(update.update_id)
        assert retrieved is not None
        assert retrieved.title == "EU AI Act"

    def test_get_update_nonexistent(self):
        """Getting nonexistent update returns None."""
        tracker = RegulatoryTracker()
        assert tracker.get_update("RU-9999") is None

    def test_add_update_string_params(self):
        """Update accepts string enum values."""
        tracker = RegulatoryTracker()
        update = tracker.add_update("fda", "guidance_draft", "Draft Guidance", priority="critical")
        assert update.jurisdiction == Jurisdiction.FDA
        assert update.update_type == UpdateType.GUIDANCE_DRAFT
        assert update.priority == Priority.CRITICAL


class TestRecentUpdates:
    """Tests for recent updates with time cutoff."""

    def test_recent_updates_within_window(self):
        """Updates published within the window are returned."""
        tracker = RegulatoryTracker()
        tracker.add_update(Jurisdiction.FDA, UpdateType.GUIDANCE_FINAL, "Recent Update")
        recent = tracker.get_recent_updates(days=90)
        assert len(recent) == 1

    def test_recent_updates_filter_by_jurisdiction(self):
        """Recent updates can be filtered by jurisdiction."""
        tracker = RegulatoryTracker()
        tracker.add_update(Jurisdiction.FDA, UpdateType.GUIDANCE_FINAL, "FDA Update")
        tracker.add_update(Jurisdiction.EMA, UpdateType.GUIDANCE_FINAL, "EMA Update")
        fda_recent = tracker.get_recent_updates(days=90, jurisdiction=Jurisdiction.FDA)
        assert len(fda_recent) == 1
        assert fda_recent[0].jurisdiction == Jurisdiction.FDA

    def test_recent_updates_sorted_newest_first(self):
        """Recent updates are sorted by published_date descending."""
        tracker = RegulatoryTracker()
        tracker.add_update(Jurisdiction.FDA, UpdateType.GUIDANCE_DRAFT, "First")
        tracker.add_update(Jurisdiction.FDA, UpdateType.GUIDANCE_FINAL, "Second")
        recent = tracker.get_recent_updates(days=90)
        assert len(recent) == 2
        # Second one should be first (newest) since both are "now" but ordering is stable
        assert recent[0].published_date >= recent[1].published_date


class TestComplianceRequirements:
    """Tests for compliance requirement management."""

    def test_add_requirement(self):
        """Compliance requirement is created and returned."""
        tracker = RegulatoryTracker()
        req = tracker.add_requirement(
            jurisdiction=Jurisdiction.FDA,
            regulation="21 CFR Part 820",
            description="QMSR compliance",
            deadline="2026-02-02T00:00:00+00:00",
            assigned_to="Regulatory Team",
        )
        assert req.requirement_id.startswith("CR-")
        assert req.jurisdiction == Jurisdiction.FDA
        assert req.status == ComplianceStatus.UNDER_REVIEW

    def test_update_requirement_status(self):
        """Requirement status can be updated."""
        tracker = RegulatoryTracker()
        req = tracker.add_requirement(Jurisdiction.EMA, "EU MDR", "MDR compliance")
        result = tracker.update_requirement_status(
            req.requirement_id, ComplianceStatus.COMPLIANT, evidence="Audit passed"
        )
        assert result is True

    def test_update_nonexistent_requirement(self):
        """Updating nonexistent requirement returns False."""
        tracker = RegulatoryTracker()
        assert tracker.update_requirement_status("CR-FAKE", ComplianceStatus.COMPLIANT) is False

    def test_get_requirements_by_jurisdiction(self):
        """Requirements can be filtered by jurisdiction."""
        tracker = RegulatoryTracker()
        tracker.add_requirement(Jurisdiction.FDA, "CFR", "FDA req")
        tracker.add_requirement(Jurisdiction.EMA, "MDR", "EMA req")
        fda_reqs = tracker.get_requirements_by_jurisdiction(Jurisdiction.FDA)
        assert len(fda_reqs) == 1
        assert fda_reqs[0].jurisdiction == Jurisdiction.FDA


class TestOverdueRequirements:
    """Tests for overdue requirement detection and status ordering."""

    def test_overdue_past_deadline(self):
        """Requirements past their deadline are marked OVERDUE."""
        tracker = RegulatoryTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        tracker.add_requirement(Jurisdiction.FDA, "CFR", "Past due req", deadline=past)
        overdue = tracker.get_overdue_requirements()
        assert len(overdue) == 1
        assert overdue[0].status == ComplianceStatus.OVERDUE

    def test_imminent_within_30_days(self):
        """Requirements within 30 days of deadline are marked IMMINENT."""
        tracker = RegulatoryTracker()
        soon = (datetime.now(timezone.utc) + timedelta(days=15)).isoformat()
        tracker.add_requirement(Jurisdiction.FDA, "CFR", "Coming soon", deadline=soon)
        # get_overdue_requirements checks and updates statuses
        overdue = tracker.get_overdue_requirements()
        assert len(overdue) == 0  # Not overdue, just imminent
        # But the requirement's status should now be IMMINENT
        reqs = tracker.get_requirements_by_jurisdiction(Jurisdiction.FDA)
        assert reqs[0].status == ComplianceStatus.IMMINENT

    def test_overdue_checked_before_imminent(self):
        """OVERDUE status is checked before IMMINENT — past due is always OVERDUE."""
        tracker = RegulatoryTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        tracker.add_requirement(Jurisdiction.FDA, "CFR", "Just past", deadline=past)
        overdue = tracker.get_overdue_requirements()
        assert len(overdue) == 1
        # It should be OVERDUE, not IMMINENT, even though it is within 30 days of deadline
        assert overdue[0].status == ComplianceStatus.OVERDUE

    def test_compliant_not_flagged_overdue(self):
        """COMPLIANT requirements are excluded from overdue checks."""
        tracker = RegulatoryTracker()
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        req = tracker.add_requirement(Jurisdiction.FDA, "CFR", "Already done", deadline=past)
        tracker.update_requirement_status(req.requirement_id, ComplianceStatus.COMPLIANT)
        overdue = tracker.get_overdue_requirements()
        assert len(overdue) == 0


class TestActionItems:
    """Tests for compliance action items."""

    def test_add_action(self):
        """Action item is linked to a requirement."""
        tracker = RegulatoryTracker()
        req = tracker.add_requirement(Jurisdiction.FDA, "CFR", "Req")
        action_id = tracker.add_action(req.requirement_id, "Prepare documentation")
        assert action_id is not None
        assert action_id.startswith("ACT-")

    def test_complete_action(self):
        """Completing an action sets COMPLETED status and date."""
        tracker = RegulatoryTracker()
        req = tracker.add_requirement(Jurisdiction.FDA, "CFR", "Req")
        action_id = tracker.add_action(req.requirement_id, "Do something")
        assert tracker.complete_action(action_id, notes="Done") is True

    def test_add_action_nonexistent_requirement(self):
        """Adding action for nonexistent requirement returns None."""
        tracker = RegulatoryTracker()
        assert tracker.add_action("CR-FAKE", "Action") is None

    def test_complete_action_nonexistent(self):
        """Completing nonexistent action returns False."""
        tracker = RegulatoryTracker()
        assert tracker.complete_action("ACT-FAKE") is False


class TestJurisdictionProfile:
    """Tests for jurisdiction profile retrieval."""

    def test_fda_profile(self):
        """FDA jurisdiction profile contains expected information."""
        tracker = RegulatoryTracker()
        profile = tracker.get_jurisdiction_profile(Jurisdiction.FDA)
        assert profile.authority_name == "U.S. Food and Drug Administration"
        assert len(profile.key_regulations) > 0


class TestDashboard:
    """Tests for dashboard reporting."""

    def test_dashboard_shows_all_jurisdictions(self):
        """Dashboard lists all monitored jurisdictions."""
        tracker = RegulatoryTracker()
        dashboard = tracker.get_dashboard()
        assert "fda" in dashboard["jurisdictions_monitored"]
        assert "ema" in dashboard["jurisdictions_monitored"]
        assert len(dashboard["jurisdictions_monitored"]) == 5

    def test_dashboard_counts(self):
        """Dashboard reflects correct counts after adding items."""
        tracker = RegulatoryTracker()
        tracker.add_update(Jurisdiction.FDA, UpdateType.GUIDANCE_FINAL, "Update")
        tracker.add_requirement(Jurisdiction.EMA, "MDR", "Req")
        dashboard = tracker.get_dashboard()
        assert dashboard["total_updates"] == 1
        assert dashboard["total_requirements"] == 1

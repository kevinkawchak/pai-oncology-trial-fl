"""Tests for regulatory/irb-management/irb_protocol_manager.py — IRB protocol.

Covers IRBProtocolManager creation, protocol lifecycle, amendment
management (including version bumping), consent versioning,
continuing reviews, and compliance summary.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "irb_protocol_manager",
    "regulatory/irb-management/irb_protocol_manager.py",
)

IRBProtocolManager = mod.IRBProtocolManager
ProtocolStatus = mod.ProtocolStatus
ReviewType = mod.ReviewType
AmendmentType = mod.AmendmentType
ConsentType = mod.ConsentType
RiskLevel = mod.RiskLevel
IRBProtocol = mod.IRBProtocol


class TestIRBProtocolManagerCreation:
    """Tests for IRBProtocolManager initialization."""

    def test_default_creation(self):
        """Newly created manager has no protocols."""
        mgr = IRBProtocolManager()
        assert mgr.list_protocols() == []

    def test_get_protocol_none(self):
        """Getting a nonexistent protocol returns None."""
        mgr = IRBProtocolManager()
        assert mgr.get_protocol("FAKE") is None


class TestProtocolCreation:
    """Tests for creating IRB protocols."""

    def test_create_protocol_basic(self):
        """Protocol is created with DRAFT status."""
        mgr = IRBProtocolManager()
        proto = mgr.create_protocol("PRO-001", "NSCLC Federated Trial")
        assert proto.protocol_id == "PRO-001"
        assert proto.title == "NSCLC Federated Trial"
        assert proto.status == ProtocolStatus.DRAFT
        assert proto.version == "1.0"

    def test_create_protocol_full_params(self):
        """Protocol accepts all optional parameters."""
        mgr = IRBProtocolManager()
        proto = mgr.create_protocol(
            "PRO-002",
            "Trial Title",
            pi_name="Dr. Smith",
            pi_institution="Test Hospital",
            review_type=ReviewType.EXPEDITED,
            risk_level=RiskLevel.GREATER_THAN_MINIMAL,
            enrollment_target=300,
            sites=["SITE-A", "SITE-B"],
        )
        assert proto.pi_name == "Dr. Smith"
        assert proto.review_type == ReviewType.EXPEDITED
        assert proto.risk_level == RiskLevel.GREATER_THAN_MINIMAL
        assert proto.enrollment_target == 300
        assert len(proto.sites) == 2

    def test_create_protocol_has_created_at(self):
        """Created protocol has a non-empty created_at timestamp."""
        mgr = IRBProtocolManager()
        proto = mgr.create_protocol("PRO-003", "Title")
        assert proto.created_at != ""


class TestProtocolLifecycle:
    """Tests for protocol lifecycle transitions."""

    def test_submit_protocol(self):
        """Submitting sets status to SUBMITTED with submission_date."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        assert mgr.submit_protocol("PRO-001", irb_name="Central IRB") is True
        proto = mgr.get_protocol("PRO-001")
        assert proto.status == ProtocolStatus.SUBMITTED
        assert proto.submission_date != ""
        assert proto.irb_name == "Central IRB"

    def test_approve_protocol(self):
        """Approving sets APPROVED status with approval/expiration dates."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        mgr.submit_protocol("PRO-001")
        assert mgr.approve_protocol("PRO-001", approval_period_days=365) is True
        proto = mgr.get_protocol("PRO-001")
        assert proto.status == ProtocolStatus.APPROVED
        assert proto.approval_date != ""
        assert proto.expiration_date != ""

    def test_approve_with_conditions(self):
        """Approving with conditions sets APPROVED_WITH_CONDITIONS."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        mgr.approve_protocol("PRO-001", conditions="Modify consent language")
        proto = mgr.get_protocol("PRO-001")
        assert proto.status == ProtocolStatus.APPROVED_WITH_CONDITIONS

    def test_is_active_property(self):
        """is_active is True only for APPROVED or APPROVED_WITH_CONDITIONS."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        proto = mgr.get_protocol("PRO-001")
        assert proto.is_active is False  # DRAFT
        mgr.approve_protocol("PRO-001")
        proto = mgr.get_protocol("PRO-001")
        assert proto.is_active is True

    def test_update_status(self):
        """Status can be explicitly updated."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        assert mgr.update_status("PRO-001", ProtocolStatus.SUSPENDED, reason="Safety concern") is True
        proto = mgr.get_protocol("PRO-001")
        assert proto.status == ProtocolStatus.SUSPENDED

    def test_lifecycle_nonexistent_protocol(self):
        """Operations on nonexistent protocol return False/None."""
        mgr = IRBProtocolManager()
        assert mgr.submit_protocol("FAKE") is False
        assert mgr.approve_protocol("FAKE") is False
        assert mgr.update_status("FAKE", ProtocolStatus.CLOSED) is False


class TestAmendmentManagement:
    """Tests for protocol amendments."""

    def test_submit_amendment(self):
        """Amendment is submitted and tracked on the protocol."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        amd_id = mgr.submit_amendment(
            "PRO-001",
            AmendmentType.MINOR,
            description="Update inclusion criteria",
        )
        assert amd_id is not None
        assert amd_id.startswith("PRO-001-AMD-")

    def test_approve_minor_amendment_bumps_minor_version(self):
        """Approving a MINOR amendment increments the minor version."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        amd_id = mgr.submit_amendment("PRO-001", AmendmentType.MINOR, "Minor change")
        mgr.approve_amendment("PRO-001", amd_id)
        proto = mgr.get_protocol("PRO-001")
        assert proto.version == "1.1"

    def test_approve_major_amendment_bumps_major_version(self):
        """Approving a MAJOR amendment increments the major version."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        amd_id = mgr.submit_amendment("PRO-001", AmendmentType.MAJOR, "Major change")
        mgr.approve_amendment("PRO-001", amd_id)
        proto = mgr.get_protocol("PRO-001")
        assert proto.version == "2.0"

    def test_approve_safety_amendment_bumps_major_version(self):
        """Approving a SAFETY amendment also increments the major version."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        amd_id = mgr.submit_amendment("PRO-001", AmendmentType.SAFETY, "Safety update")
        mgr.approve_amendment("PRO-001", amd_id)
        proto = mgr.get_protocol("PRO-001")
        assert proto.version == "2.0"

    def test_submit_amendment_nonexistent(self):
        """Submitting amendment for nonexistent protocol returns None."""
        mgr = IRBProtocolManager()
        assert mgr.submit_amendment("FAKE", AmendmentType.MINOR, "desc") is None


class TestConsentVersioning:
    """Tests for consent version management."""

    def test_add_consent_version(self):
        """Consent version is added to the protocol."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        vid = mgr.add_consent_version(
            "PRO-001",
            ConsentType.FULL_CONSENT,
            ai_ml_disclosures=["AI-assisted diagnosis used"],
        )
        assert vid is not None
        proto = mgr.get_protocol("PRO-001")
        assert len(proto.consent_versions) == 1

    def test_auto_version_number(self):
        """Version number is auto-incremented when not provided."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        mgr.add_consent_version("PRO-001", ConsentType.FULL_CONSENT)
        mgr.add_consent_version("PRO-001", ConsentType.FULL_CONSENT)
        proto = mgr.get_protocol("PRO-001")
        versions = [cv.version_number for cv in proto.consent_versions]
        assert versions == ["1.0", "2.0"]

    def test_approve_consent(self):
        """Approving consent sets approved_date and effective_date."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial")
        vid = mgr.add_consent_version("PRO-001", ConsentType.FULL_CONSENT)
        assert mgr.approve_consent("PRO-001", vid) is True
        proto = mgr.get_protocol("PRO-001")
        cv = proto.consent_versions[0]
        assert cv.approved_date != ""
        assert cv.effective_date != ""


class TestComplianceSummary:
    """Tests for compliance summary generation."""

    def test_compliance_summary_basic(self):
        """Compliance summary contains expected fields."""
        mgr = IRBProtocolManager()
        mgr.create_protocol("PRO-001", "Trial", enrollment_target=200)
        summary = mgr.get_compliance_summary("PRO-001")
        assert summary["protocol_id"] == "PRO-001"
        assert summary["status"] == "draft"
        assert summary["version"] == "1.0"
        assert summary["is_active"] is False
        assert summary["amendments"] == 0

    def test_compliance_summary_nonexistent(self):
        """Compliance summary for nonexistent protocol returns error."""
        mgr = IRBProtocolManager()
        summary = mgr.get_compliance_summary("FAKE")
        assert "error" in summary

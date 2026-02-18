"""Tests for federated/site_enrollment.py — SiteEnrollmentManager.

Covers creation, enroll site, get enrolled sites, audit trail returns
copy, status transitions (activate, suspend, withdraw, reactivate),
quality-based site selection, and enrollment summary.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import PROJECT_ROOT, load_module

mod = load_module("federated.site_enrollment", "federated/site_enrollment.py")

SiteEnrollmentManager = mod.SiteEnrollmentManager
SiteProfile = mod.SiteProfile
SiteStatus = mod.SiteStatus


def _make_manager(**kwargs) -> SiteEnrollmentManager:
    kwargs.setdefault("min_patients_per_site", 10)
    return SiteEnrollmentManager(study_id="STUDY-001", **kwargs)


def _enroll_and_activate(mgr: SiteEnrollmentManager, site_id: str, patients: int = 50) -> SiteProfile:
    profile = mgr.enroll_site(site_id, f"Hospital {site_id}", patient_count=patients)
    mgr.mark_data_ready(site_id)
    mgr.mark_compliance_passed(site_id)
    mgr.activate_site(site_id)
    return mgr.get_site(site_id)


class TestSiteEnrollmentManagerCreation:
    """Tests for manager instantiation."""

    def test_creation_defaults(self):
        mgr = _make_manager()
        assert mgr.study_id == "STUDY-001"
        assert mgr.min_patients_per_site == 10
        assert mgr.min_quality_score == 0.3

    def test_no_sites_initially(self):
        mgr = _make_manager()
        summary = mgr.get_enrollment_summary()
        assert summary["total_sites"] == 0
        assert summary["total_patients"] == 0


class TestEnrollSite:
    """Tests for site enrollment."""

    def test_enroll_site_creates_profile(self):
        mgr = _make_manager()
        profile = mgr.enroll_site("SITE-A", "Hospital Alpha", patient_count=50)
        assert profile.site_id == "SITE-A"
        assert profile.institution_name == "Hospital Alpha"
        assert profile.status == SiteStatus.PENDING
        assert profile.patient_count == 50
        assert profile.enrolled_at is not None

    def test_enroll_site_with_capabilities(self):
        mgr = _make_manager()
        profile = mgr.enroll_site("SITE-B", "Hospital Beta", capabilities=["imaging", "biopsy"])
        assert "imaging" in profile.capabilities
        assert "biopsy" in profile.capabilities

    def test_enroll_multiple_sites(self):
        mgr = _make_manager()
        mgr.enroll_site("S1", "H1")
        mgr.enroll_site("S2", "H2")
        mgr.enroll_site("S3", "H3")
        summary = mgr.get_enrollment_summary()
        assert summary["total_sites"] == 3

    def test_get_site_returns_profile(self):
        mgr = _make_manager()
        mgr.enroll_site("SITE-X", "Hospital X")
        profile = mgr.get_site("SITE-X")
        assert profile is not None
        assert profile.site_id == "SITE-X"

    def test_get_site_nonexistent_returns_none(self):
        mgr = _make_manager()
        assert mgr.get_site("NONEXISTENT") is None


class TestStatusTransitions:
    """Tests for lifecycle state machine transitions."""

    def test_activate_site_success(self):
        mgr = _make_manager()
        mgr.enroll_site("S1", "H1", patient_count=20)
        mgr.mark_data_ready("S1")
        mgr.mark_compliance_passed("S1")
        result = mgr.activate_site("S1")
        assert result is True
        assert mgr.get_site("S1").status == SiteStatus.ACTIVE

    def test_activate_fails_without_data_ready(self):
        mgr = _make_manager()
        mgr.enroll_site("S1", "H1", patient_count=20)
        mgr.mark_compliance_passed("S1")
        assert mgr.activate_site("S1") is False

    def test_activate_fails_without_compliance(self):
        mgr = _make_manager()
        mgr.enroll_site("S1", "H1", patient_count=20)
        mgr.mark_data_ready("S1")
        assert mgr.activate_site("S1") is False

    def test_activate_fails_insufficient_patients(self):
        mgr = _make_manager(min_patients_per_site=100)
        mgr.enroll_site("S1", "H1", patient_count=5)
        mgr.mark_data_ready("S1")
        mgr.mark_compliance_passed("S1")
        assert mgr.activate_site("S1") is False

    def test_activate_nonexistent_site_returns_false(self):
        mgr = _make_manager()
        assert mgr.activate_site("GHOST") is False

    def test_suspend_site(self):
        mgr = _make_manager()
        _enroll_and_activate(mgr, "S1")
        result = mgr.suspend_site("S1", reason="data issue")
        assert result is True
        assert mgr.get_site("S1").status == SiteStatus.SUSPENDED

    def test_suspend_nonexistent_returns_false(self):
        mgr = _make_manager()
        assert mgr.suspend_site("GHOST") is False

    def test_withdraw_site(self):
        mgr = _make_manager()
        _enroll_and_activate(mgr, "S1")
        result = mgr.withdraw_site("S1", reason="PI request")
        assert result is True
        assert mgr.get_site("S1").status == SiteStatus.WITHDRAWN

    def test_reactivate_suspended_site(self):
        mgr = _make_manager()
        _enroll_and_activate(mgr, "S1")
        mgr.suspend_site("S1")
        result = mgr.reactivate_site("S1")
        assert result is True
        assert mgr.get_site("S1").status == SiteStatus.ACTIVE

    def test_reactivate_non_suspended_fails(self):
        mgr = _make_manager()
        _enroll_and_activate(mgr, "S1")
        # Site is ACTIVE, not SUSPENDED
        assert mgr.reactivate_site("S1") is False


class TestSiteSelectionAndQuality:
    """Tests for quality-based site selection."""

    def test_get_active_sites(self):
        mgr = _make_manager()
        _enroll_and_activate(mgr, "S1")
        _enroll_and_activate(mgr, "S2")
        mgr.enroll_site("S3", "H3")  # stays PENDING
        active = mgr.get_active_sites()
        assert len(active) == 2

    def test_select_sites_sorted_by_quality(self):
        mgr = _make_manager()
        _enroll_and_activate(mgr, "S1")
        _enroll_and_activate(mgr, "S2")
        mgr.update_quality_score("S1", 0.5)
        mgr.update_quality_score("S2", 0.9)
        selected = mgr.select_sites_for_round()
        assert selected[0].site_id == "S2"

    def test_select_sites_max_sites(self):
        mgr = _make_manager()
        for i in range(5):
            _enroll_and_activate(mgr, f"S{i}")
        selected = mgr.select_sites_for_round(max_sites=2)
        assert len(selected) == 2

    def test_quality_score_clamped(self):
        mgr = _make_manager()
        mgr.enroll_site("S1", "H1")
        mgr.update_quality_score("S1", 1.5)
        assert mgr.get_site("S1").quality_score == 1.0
        mgr.update_quality_score("S1", -0.5)
        assert mgr.get_site("S1").quality_score == 0.0


class TestAuditTrail:
    """Tests for audit trail."""

    def test_audit_trail_populated(self):
        mgr = _make_manager()
        mgr.enroll_site("S1", "H1")
        trail = mgr.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["event"] == "site_enrolled"
        assert trail[0]["site_id"] == "S1"

    def test_audit_trail_returns_copy(self):
        mgr = _make_manager()
        mgr.enroll_site("S1", "H1")
        trail = mgr.get_audit_trail()
        trail.append({"event": "fake"})
        assert len(mgr.get_audit_trail()) == 1

    def test_record_round_participation(self):
        mgr = _make_manager()
        _enroll_and_activate(mgr, "S1")
        mgr.record_round_participation("S1")
        mgr.record_round_participation("S1")
        assert mgr.get_site("S1").rounds_participated == 2
        assert mgr.get_site("S1").last_active is not None

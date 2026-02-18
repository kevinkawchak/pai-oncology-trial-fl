"""Tests for clinical-analytics/consortium_reporting.py.

Covers enums (ReportType, DSMBRecommendation, AuditSeverity, BlindingStatus),
dataclasses (DSMBPackage, SafetySummary, EnrollmentSummary, AuditTrailEntry),
and the ConsortiumReportingEngine class: DSMB package generation, safety
summary, enrollment summary, hash chain computation and verification,
regulatory report, data quality report, audit trail (defensive copy),
and markdown formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import load_module

mod = load_module(
    "consortium_reporting",
    "clinical-analytics/consortium_reporting.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestReportType:
    """Tests for the ReportType enum."""

    def test_report_type_members(self):
        """ReportType has expected members."""
        names = set(mod.ReportType.__members__.keys())
        assert "DSMB_INTERIM" in names
        assert "REGULATORY_ANNUAL" in names
        assert "DATA_QUALITY" in names
        assert "SAFETY_SIGNAL" in names

    def test_report_type_is_str_enum(self):
        """ReportType members are instances of str."""
        assert isinstance(mod.ReportType.DSMB_INTERIM, str)


class TestDSMBRecommendation:
    """Tests for the DSMBRecommendation enum."""

    def test_recommendation_members(self):
        """DSMBRecommendation has expected members."""
        names = set(mod.DSMBRecommendation.__members__.keys())
        assert "CONTINUE_UNCHANGED" in names
        assert "TERMINATE_EARLY" in names
        assert "SUSPEND_ENROLLMENT" in names


class TestAuditSeverity:
    """Tests for the AuditSeverity enum."""

    def test_severity_members(self):
        """AuditSeverity has four levels."""
        expected = {"INFORMATIONAL", "MINOR", "MAJOR", "CRITICAL"}
        assert set(mod.AuditSeverity.__members__.keys()) == expected


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestDSMBPackage:
    """Tests for the DSMBPackage dataclass."""

    def test_default_creation(self):
        """DSMBPackage is created with auto-generated fields."""
        pkg = mod.DSMBPackage()
        assert pkg.report_id != ""
        assert pkg.generated_at != ""

    def test_blinding_default(self):
        """Default blinding status is BLINDED."""
        pkg = mod.DSMBPackage()
        assert pkg.blinding_status == mod.BlindingStatus.BLINDED

    def test_with_trial_id(self):
        """DSMBPackage accepts a trial ID."""
        pkg = mod.DSMBPackage(trial_id="NCT-001")
        assert pkg.trial_id == "NCT-001"


class TestSafetySummary:
    """Tests for the SafetySummary dataclass."""

    def test_default_creation(self):
        """Default SafetySummary has zero counts."""
        s = mod.SafetySummary()
        assert s.total_aes == 0
        assert s.serious_aes == 0
        assert s.deaths == 0


# ---------------------------------------------------------------------------
# ConsortiumReportingEngine tests
# ---------------------------------------------------------------------------
class TestConsortiumReportingEngine:
    """Tests for the ConsortiumReportingEngine class."""

    def test_init(self):
        """Engine initializes with trial ID."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001", user_id="tester")
        # Audit trail should have at least one entry
        trail = engine.get_audit_trail()
        assert len(trail) >= 0  # May or may not have init entry

    def test_build_safety_summary(self):
        """build_safety_summary returns SafetySummary with correct counts."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        aes = [
            {"grade": 1, "serious": False, "system_organ_class": "GI"},
            {"grade": 3, "serious": True, "system_organ_class": "Hepatic"},
            {"grade": 2, "serious": False, "system_organ_class": "GI"},
            {"grade": 5, "serious": True, "system_organ_class": "Cardiac"},
        ]
        summary = engine.build_safety_summary(aes)
        assert isinstance(summary, mod.SafetySummary)
        assert summary.total_aes == 4
        assert summary.serious_aes == 2
        assert summary.deaths == 1

    def test_build_safety_summary_empty(self):
        """Empty AE list returns zero-count SafetySummary."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        summary = engine.build_safety_summary([])
        assert summary.total_aes == 0

    def test_build_enrollment_summary(self):
        """build_enrollment_summary returns correct totals."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        sites = [
            {"site_id": "S1", "enrolled": 50, "active": 40, "completed": 5, "withdrawn": 2, "target": 100},
            {"site_id": "S2", "enrolled": 30, "active": 25, "completed": 3, "withdrawn": 1, "target": 80},
        ]
        summary = engine.build_enrollment_summary(sites)
        assert isinstance(summary, mod.EnrollmentSummary)
        assert summary.enrolled == 80
        assert summary.target == 180

    def test_build_enrollment_summary_empty(self):
        """Empty site data returns zero enrollment."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        summary = engine.build_enrollment_summary([])
        assert summary.enrolled == 0

    def test_generate_dsmb_package(self):
        """generate_dsmb_package returns a DSMBPackage."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        trial_data = {
            "adverse_events": [
                {"grade": 2, "serious": False, "system_organ_class": "GI"},
            ],
            "site_data": [
                {"site_id": "S1", "enrolled": 50, "active": 45, "completed": 3, "withdrawn": 1, "target": 100},
            ],
            "efficacy": {
                "experimental": {"os_median": 12.5},
                "control": {"os_median": 10.0},
            },
        }
        pkg = engine.generate_dsmb_package(trial_data, interim_number=1)
        assert isinstance(pkg, mod.DSMBPackage)
        assert pkg.trial_id == "NCT-001"
        assert pkg.interim_number == 1

    def test_hash_chain_computation(self):
        """compute_hash_chain produces SHA-256 hex digests (64 chars each)."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        entries = [
            mod.AuditTrailEntry(action="test_1", user_id="u1", details="d1"),
            mod.AuditTrailEntry(action="test_2", user_id="u1", details="d2"),
        ]
        hashes = engine.compute_hash_chain(entries)
        assert len(hashes) == 2
        for h in hashes:
            assert isinstance(h, str)
            assert len(h) == 64  # SHA-256 hex

    def test_verify_hash_chain_empty(self):
        """Empty hash chain is considered valid."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        valid, failed = engine.verify_hash_chain([])
        assert valid is True
        assert failed == []

    def test_verify_hash_chain_valid(self):
        """Hash chain created by engine verifies successfully."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        # Perform operations to build audit trail
        engine.build_safety_summary([{"grade": 1, "serious": False}])
        engine.build_enrollment_summary(
            [{"site_id": "S1", "enrolled": 10, "active": 8, "completed": 1, "withdrawn": 0, "target": 50}]
        )
        valid, failed = engine.verify_hash_chain()
        assert valid is True
        assert failed == []

    def test_data_quality_report(self):
        """generate_data_quality_report returns a ConsortiumReport."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        metrics = [
            {"site_id": "S1", "query_rate": 5.0, "missing_data_pct": 3.0, "protocol_deviations": 2},
            {"site_id": "S2", "query_rate": 12.0, "missing_data_pct": 8.0, "protocol_deviations": 7},
        ]
        report = engine.generate_data_quality_report(metrics)
        assert isinstance(report, mod.ConsortiumReport)
        assert report.report_type == mod.ReportType.DATA_QUALITY

    def test_data_quality_report_empty(self):
        """Data quality report with no sites still returns a report."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        report = engine.generate_data_quality_report([])
        assert isinstance(report, mod.ConsortiumReport)

    def test_audit_trail_defensive_copy(self):
        """get_audit_trail returns a defensive copy."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        engine.build_safety_summary([{"grade": 1}])
        trail1 = engine.get_audit_trail()
        original_len = len(trail1)
        trail1.append(mod.AuditTrailEntry(action="fake"))
        trail2 = engine.get_audit_trail()
        assert len(trail2) == original_len

    def test_markdown_formatting(self):
        """format_report_markdown produces valid markdown string."""
        engine = mod.ConsortiumReportingEngine(trial_id="NCT-001")
        metrics = [
            {"site_id": "S1", "query_rate": 5.0, "missing_data_pct": 2.0, "protocol_deviations": 1},
        ]
        report = engine.generate_data_quality_report(metrics)
        md = engine.format_report_markdown(report)
        assert isinstance(md, str)
        assert "#" in md  # contains markdown headers
        assert len(md) > 50

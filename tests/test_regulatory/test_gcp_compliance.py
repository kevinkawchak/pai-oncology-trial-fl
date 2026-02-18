"""Tests for regulatory/ich-gcp/gcp_compliance_checker.py — GCP compliance.

Covers GCPComplianceChecker creation, principle enumeration, compliance
assessment, score calculation (excluding NOT_ASSESSED), all compliance
levels, finding categorization, and report generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "gcp_compliance_checker",
    "regulatory/ich-gcp/gcp_compliance_checker.py",
)

GCPComplianceChecker = mod.GCPComplianceChecker
GCPPrinciple = mod.GCPPrinciple
ComplianceLevel = mod.ComplianceLevel
FindingCategory = mod.FindingCategory
TrialPhase = mod.TrialPhase
ComplianceReport = mod.ComplianceReport
ComplianceFinding = mod.ComplianceFinding


class TestGCPPrincipleEnum:
    """Tests for GCPPrinciple enum."""

    def test_principle_count(self):
        """There should be 13 GCP principles per ICH E6(R3)."""
        assert len(GCPPrinciple) == 13

    def test_key_principles_exist(self):
        """Key principles are defined."""
        assert GCPPrinciple.ETHICAL_CONDUCT.value == "ethical_conduct"
        assert GCPPrinciple.INFORMED_CONSENT.value == "informed_consent"
        assert GCPPrinciple.DATA_INTEGRITY.value == "data_integrity"
        assert GCPPrinciple.RISK_PROPORTIONATE.value == "risk_proportionate_approach"


class TestComplianceLevelEnum:
    """Tests for ComplianceLevel enum."""

    def test_all_levels_exist(self):
        """All five compliance levels are defined."""
        assert ComplianceLevel.COMPLIANT.value == "compliant"
        assert ComplianceLevel.MINOR_FINDING.value == "minor_finding"
        assert ComplianceLevel.MAJOR_FINDING.value == "major_finding"
        assert ComplianceLevel.CRITICAL_FINDING.value == "critical_finding"
        assert ComplianceLevel.NOT_ASSESSED.value == "not_assessed"


class TestGCPComplianceCheckerCreation:
    """Tests for GCPComplianceChecker initialization."""

    def test_default_creation(self):
        """Newly created checker has a checklist."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        assert isinstance(checklist, list)
        assert len(checklist) > 0

    def test_checklist_returns_copy(self):
        """get_checklist returns a copy, not the internal list."""
        checker = GCPComplianceChecker()
        c1 = checker.get_checklist()
        c2 = checker.get_checklist()
        assert c1 is not c2


class TestAssessment:
    """Tests for trial assessment."""

    def test_assess_trial_no_results(self):
        """Assessment without check_results marks everything NOT_ASSESSED."""
        checker = GCPComplianceChecker()
        report = checker.assess_trial("TRIAL-001")
        assert report.trial_id == "TRIAL-001"
        assert report.trial_phase == TrialPhase.PHASE_III
        # All NOT_ASSESSED means score is 0.0
        np.testing.assert_allclose(report.overall_score, 0.0)

    def test_assess_trial_all_compliant(self):
        """All COMPLIANT checks produce score of 1.0."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        results = {}
        for area in checklist:
            for check_text in area["checks"]:
                results[check_text] = ComplianceLevel.COMPLIANT
        report = checker.assess_trial("TRIAL-002", check_results=results)
        np.testing.assert_allclose(report.overall_score, 1.0)
        assert len(report.findings) == 0

    def test_assess_trial_all_critical(self):
        """All CRITICAL_FINDING checks produce score of 0.0."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        results = {}
        for area in checklist:
            for check_text in area["checks"]:
                results[check_text] = ComplianceLevel.CRITICAL_FINDING
        report = checker.assess_trial("TRIAL-003", check_results=results)
        np.testing.assert_allclose(report.overall_score, 0.0)
        assert len(report.findings) > 0

    def test_assess_trial_mixed_scores(self):
        """Mixed compliance levels produce intermediate score."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        first_check = checklist[0]["checks"][0]
        second_check = checklist[0]["checks"][1] if len(checklist[0]["checks"]) > 1 else None

        results = {first_check: ComplianceLevel.COMPLIANT}
        if second_check:
            results[second_check] = ComplianceLevel.MINOR_FINDING

        report = checker.assess_trial("TRIAL-004", check_results=results)
        # Score should be between 0 and 1 exclusive
        assert 0.0 < report.overall_score < 1.0

    def test_score_excludes_not_assessed(self):
        """NOT_ASSESSED items are excluded from the score denominator."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        # Only assess one check as COMPLIANT, rest are NOT_ASSESSED
        first_check = checklist[0]["checks"][0]
        results = {first_check: ComplianceLevel.COMPLIANT}
        report = checker.assess_trial("TRIAL-005", check_results=results)
        # With only one COMPLIANT check assessed, score should be 1.0
        np.testing.assert_allclose(report.overall_score, 1.0)


class TestScoreCalculation:
    """Tests for mathematical correctness of score calculation."""

    def test_score_minor_finding_value(self):
        """MINOR_FINDING scores 0.7 per check."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        first_check = checklist[0]["checks"][0]
        results = {first_check: ComplianceLevel.MINOR_FINDING}
        report = checker.assess_trial("TRIAL-006", check_results=results)
        np.testing.assert_allclose(report.overall_score, 0.7)

    def test_score_major_finding_value(self):
        """MAJOR_FINDING scores 0.3 per check."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        first_check = checklist[0]["checks"][0]
        results = {first_check: ComplianceLevel.MAJOR_FINDING}
        report = checker.assess_trial("TRIAL-007", check_results=results)
        np.testing.assert_allclose(report.overall_score, 0.3)

    def test_score_average_of_two_checks(self):
        """Score averages across assessed checks: (1.0 + 0.7) / 2 = 0.85."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        check1 = checklist[0]["checks"][0]
        check2 = checklist[0]["checks"][1] if len(checklist[0]["checks"]) > 1 else checklist[1]["checks"][0]
        results = {
            check1: ComplianceLevel.COMPLIANT,
            check2: ComplianceLevel.MINOR_FINDING,
        }
        report = checker.assess_trial("TRIAL-008", check_results=results)
        np.testing.assert_allclose(report.overall_score, 0.85)


class TestFindings:
    """Tests for compliance findings."""

    def test_findings_generated_for_non_compliant(self):
        """Findings are created for MINOR, MAJOR, CRITICAL levels."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        checks = checklist[0]["checks"]
        results = {}
        if len(checks) >= 3:
            results[checks[0]] = ComplianceLevel.MINOR_FINDING
            results[checks[1]] = ComplianceLevel.MAJOR_FINDING
            results[checks[2]] = ComplianceLevel.CRITICAL_FINDING
        report = checker.assess_trial("TRIAL-009", check_results=results)
        assert len(report.findings) >= min(3, len(checks))

    def test_no_findings_for_compliant(self):
        """COMPLIANT checks do not generate findings."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        first_check = checklist[0]["checks"][0]
        results = {first_check: ComplianceLevel.COMPLIANT}
        report = checker.assess_trial("TRIAL-010", check_results=results)
        assert len(report.findings) == 0

    def test_add_manual_finding(self):
        """Manual findings can be added to an existing report."""
        checker = GCPComplianceChecker()
        report = checker.assess_trial("TRIAL-011")
        fid = checker.add_finding(
            report.report_id,
            GCPPrinciple.DATA_INTEGRITY,
            ComplianceLevel.MAJOR_FINDING,
            FindingCategory.DATA_MANAGEMENT,
            description="Audit trail gaps found",
        )
        assert fid is not None
        updated_report = checker.get_report(report.report_id)
        assert len(updated_report.findings) == 1

    def test_add_finding_nonexistent_report(self):
        """Adding finding to nonexistent report returns None."""
        checker = GCPComplianceChecker()
        assert (
            checker.add_finding(
                "FAKE-REPORT",
                GCPPrinciple.DATA_INTEGRITY,
                ComplianceLevel.MINOR_FINDING,
                FindingCategory.DOCUMENTATION,
                "test",
            )
            is None
        )


class TestReportContent:
    """Tests for report summary and recommendations."""

    def test_report_has_summary(self):
        """Report has a non-empty summary string."""
        checker = GCPComplianceChecker()
        report = checker.assess_trial("TRIAL-012")
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_report_has_recommendations(self):
        """Report has at least one recommendation."""
        checker = GCPComplianceChecker()
        report = checker.assess_trial("TRIAL-013")
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) >= 1

    def test_critical_findings_trigger_immediate_recommendation(self):
        """Critical findings generate an IMMEDIATE recommendation."""
        checker = GCPComplianceChecker()
        checklist = checker.get_checklist()
        first_check = checklist[0]["checks"][0]
        results = {first_check: ComplianceLevel.CRITICAL_FINDING}
        report = checker.assess_trial("TRIAL-014", check_results=results)
        immediate = [r for r in report.recommendations if "IMMEDIATE" in r]
        assert len(immediate) >= 1

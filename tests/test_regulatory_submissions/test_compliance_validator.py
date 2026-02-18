"""Tests for regulatory-submissions/compliance_validator.py.

Covers enums (Regulation, ComplianceStatus, FindingSeverity, RemediationStatus,
RequirementCategory), dataclasses (RegulatoryRequirement, ComplianceFinding,
RemediationAction, ComplianceAssessment, AuditTrailEntry), and ComplianceValidator:
  - Initialization with multiple regulations
  - Requirement loading
  - Evidence submission
  - Individual requirement assessment
  - Full assessment run
  - Compliance scoring (weighted and unweighted)
  - Gap analysis by regulation
  - Remediation management
  - Audit trail
  - Compliance report generation
  - Finding filtering by severity
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
    "compliance_validator",
    "regulatory-submissions/compliance_validator.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_validator(regs=None):
    if regs is None:
        regs = [mod.Regulation.FDA_21CFR820, mod.Regulation.ISO_14971]
    return mod.ComplianceValidator("SUB-TEST", regs)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestRegulation:
    def test_members(self):
        names = set(mod.Regulation.__members__.keys())
        assert "FDA_21CFR820" in names
        assert "GDPR" in names
        assert "ISO_14971" in names
        assert "IEC_62304" in names
        assert "MDR_2017_745" in names

    def test_regulation_count(self):
        assert len(mod.Regulation) == 9

    def test_is_str(self):
        assert isinstance(mod.Regulation.FDA_21CFR820, str)


class TestComplianceStatus:
    def test_members(self):
        names = set(mod.ComplianceStatus.__members__.keys())
        assert "COMPLIANT" in names
        assert "NON_COMPLIANT" in names
        assert "PENDING_REVIEW" in names

    def test_count(self):
        assert len(mod.ComplianceStatus) == 5


class TestFindingSeverity:
    def test_members(self):
        assert mod.FindingSeverity.CRITICAL.value == "critical"
        assert mod.FindingSeverity.MAJOR.value == "major"
        assert mod.FindingSeverity.MINOR.value == "minor"
        assert mod.FindingSeverity.OBSERVATION.value == "observation"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestRegulatoryRequirement:
    def test_creation(self):
        req = mod.RegulatoryRequirement(
            regulation=mod.Regulation.FDA_21CFR820,
            section="820.30",
            title="Design Controls",
        )
        assert req.section == "820.30"
        assert req.is_mandatory is True

    def test_weight_bounds(self):
        req = mod.RegulatoryRequirement(weight=100.0)
        assert req.weight <= 10.0
        req2 = mod.RegulatoryRequirement(weight=-5.0)
        assert req2.weight >= 0.1


class TestComplianceAssessment:
    def test_compute_scores_empty(self):
        a = mod.ComplianceAssessment()
        a.compute_scores()
        assert a.unweighted_score == 0.0
        assert a.is_passing is False

    def test_compute_scores_passing(self):
        a = mod.ComplianceAssessment(compliant_count=9, non_compliant_count=1, partial_count=0)
        a.compute_scores()
        assert a.unweighted_score == 90.0
        assert a.is_passing is True


class TestAuditTrailEntry:
    def test_hash(self):
        entry = mod.AuditTrailEntry(action="test", entity_id="E1", details="d")
        h = entry.compute_hash()
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------
class TestComplianceValidator:
    def test_init(self):
        v = _make_validator()
        assert v.submission_id == "SUB-TEST"
        assert v.requirement_count > 0

    def test_init_all_regulations(self):
        v = mod.ComplianceValidator("SUB-ALL")
        assert v.requirement_count > 0

    def test_submit_evidence(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        assert len(reqs) > 0
        ok = v.submit_evidence(reqs[0].requirement_id, "DHF reference")
        assert ok is True

    def test_submit_evidence_nonexistent(self):
        v = _make_validator()
        ok = v.submit_evidence("nonexistent", "evidence")
        assert ok is False

    def test_assess_requirement(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        finding = v.assess_requirement(reqs[0].requirement_id, mod.ComplianceStatus.COMPLIANT, "evidence")
        assert finding is not None
        assert finding.status == mod.ComplianceStatus.COMPLIANT

    def test_assess_nonexistent(self):
        v = _make_validator()
        finding = v.assess_requirement("nonexistent", mod.ComplianceStatus.COMPLIANT)
        assert finding is None

    def test_run_assessment(self):
        v = _make_validator()
        assessment = v.run_assessment()
        assert assessment.total_requirements > 0
        assert assessment.summary

    def test_run_assessment_with_evidence(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        evidence = {reqs[0].requirement_id: "Evidence A", reqs[1].requirement_id: "Evidence B"}
        assessment = v.run_assessment(evidence_map=evidence)
        assert assessment.compliant_count >= 2

    def test_weighted_score(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        for req in reqs:
            v.submit_evidence(req.requirement_id, "Full evidence")
        assessment = v.run_assessment()
        assert assessment.weighted_score > 0

    def test_gap_analysis(self):
        v = _make_validator()
        v.run_assessment()
        gaps = v.generate_gap_analysis()
        assert mod.Regulation.FDA_21CFR820 in gaps
        assert "score" in gaps[mod.Regulation.FDA_21CFR820]

    def test_create_remediation(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        finding = v.assess_requirement(reqs[0].requirement_id, mod.ComplianceStatus.NON_COMPLIANT)
        rem = v.create_remediation(finding.finding_id, "Fix gap", owner="qa_lead")
        assert rem is not None
        assert rem.owner == "qa_lead"

    def test_create_remediation_nonexistent(self):
        v = _make_validator()
        rem = v.create_remediation("nonexistent", "Fix")
        assert rem is None

    def test_update_remediation_status(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        finding = v.assess_requirement(reqs[0].requirement_id, mod.ComplianceStatus.NON_COMPLIANT)
        rem = v.create_remediation(finding.finding_id, "Fix")
        ok = v.update_remediation_status(rem.action_id, mod.RemediationStatus.RESOLVED, "Done")
        assert ok is True

    def test_open_remediations(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        finding = v.assess_requirement(reqs[0].requirement_id, mod.ComplianceStatus.NON_COMPLIANT)
        v.create_remediation(finding.finding_id, "Fix")
        open_rems = v.get_open_remediations()
        assert len(open_rems) >= 1

    def test_compliance_report(self):
        v = _make_validator()
        assessment = v.run_assessment()
        report = v.generate_compliance_report(assessment)
        assert "SUB-TEST" in report
        assert "Gap Analysis" in report

    def test_findings_by_severity(self):
        v = _make_validator()
        v.run_assessment()
        observations = v.get_findings_by_severity(mod.FindingSeverity.OBSERVATION)
        assert isinstance(observations, list)

    def test_audit_trail_populated(self):
        v = _make_validator()
        reqs = v.get_requirements_by_regulation(mod.Regulation.FDA_21CFR820)
        v.submit_evidence(reqs[0].requirement_id, "ev")
        assert len(v.audit_trail) > 0

    def test_to_dict(self):
        v = _make_validator()
        d = v.to_dict()
        assert d["submission_id"] == "SUB-TEST"
        assert "requirement_count" in d

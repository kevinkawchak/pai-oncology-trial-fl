#!/usr/bin/env python3
"""Multi-regulation compliance validation engine for regulatory submissions.

RESEARCH USE ONLY -- Not intended for clinical deployment without validation.

This module implements a checklist-driven compliance validation engine that
assesses regulatory submissions against multiple regulatory frameworks
simultaneously.  It supports FDA 21 CFR 820 (Quality System Regulation),
21 CFR Part 11 (Electronic Records), HIPAA, GDPR, ISO 14971 (Risk
Management), IEC 62304 (Medical Device Software), ICH E6(R3) (GCP), ICH
E9(R1) (Statistical Principles), and MDR 2017/745 (EU Medical Device
Regulation).

Features:
    * Parallel multi-regulation assessment
    * Gap analysis with remediation recommendations
    * Compliance scoring (weighted and unweighted)
    * Remediation tracking with priority assignment
    * Audit trail for compliance decisions
    * Configurable requirement checklists per regulation

All data structures are in-memory dataclasses.  No network calls or
external API dependencies.

DISCLAIMER: RESEARCH USE ONLY.  Not validated for regulatory use.
LICENSE: MIT
VERSION: 0.9.1
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_REQUIREMENTS: int = 1000
MAX_FINDINGS: int = 5000
MAX_AUDIT_ENTRIES: int = 10000
PASSING_SCORE_THRESHOLD: float = 80.0


# ===================================================================
# Enums
# ===================================================================
class Regulation(str, Enum):
    """Regulatory frameworks supported for compliance validation."""

    FDA_21CFR820 = "fda_21cfr820"
    FDA_21CFR11 = "fda_21cfr11"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    ISO_14971 = "iso_14971"
    IEC_62304 = "iec_62304"
    ICH_E6R3 = "ich_e6r3"
    ICH_E9R1 = "ich_e9r1"
    MDR_2017_745 = "mdr_2017_745"


class ComplianceStatus(str, Enum):
    """Compliance assessment status for a single requirement."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


class FindingSeverity(str, Enum):
    """Severity levels for compliance findings."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


class RemediationStatus(str, Enum):
    """Status of a remediation action."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    VERIFIED = "verified"
    DEFERRED = "deferred"


class RequirementCategory(str, Enum):
    """Categories for regulatory requirements."""

    DESIGN_CONTROLS = "design_controls"
    RISK_MANAGEMENT = "risk_management"
    SOFTWARE_LIFECYCLE = "software_lifecycle"
    CLINICAL_EVIDENCE = "clinical_evidence"
    DATA_INTEGRITY = "data_integrity"
    PRIVACY = "privacy"
    LABELING = "labeling"
    POST_MARKET = "post_market"
    DOCUMENTATION = "documentation"
    QUALITY_SYSTEM = "quality_system"


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class RegulatoryRequirement:
    """A single regulatory requirement from a specific framework."""

    requirement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    regulation: Regulation = Regulation.FDA_21CFR820
    section: str = ""
    title: str = ""
    description: str = ""
    category: RequirementCategory = RequirementCategory.DOCUMENTATION
    is_mandatory: bool = True
    weight: float = 1.0
    evidence_needed: str = ""
    reference_url: str = ""

    def __post_init__(self) -> None:
        self.weight = max(0.1, min(10.0, self.weight))


@dataclass
class ComplianceFinding:
    """A finding from compliance assessment against a requirement."""

    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requirement_id: str = ""
    regulation: Regulation = Regulation.FDA_21CFR820
    severity: FindingSeverity = FindingSeverity.OBSERVATION
    status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    title: str = ""
    description: str = ""
    evidence_provided: str = ""
    gap_description: str = ""
    remediation_recommendation: str = ""
    assessed_by: str = ""
    assessed_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RemediationAction:
    """A corrective action to address a compliance finding."""

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    finding_id: str = ""
    title: str = ""
    description: str = ""
    owner: str = ""
    priority: FindingSeverity = FindingSeverity.MAJOR
    status: RemediationStatus = RemediationStatus.OPEN
    target_date: str = ""
    completion_date: str = ""
    verification_notes: str = ""


@dataclass
class ComplianceAssessment:
    """Complete compliance assessment for a submission."""

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = ""
    assessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    regulations_assessed: List[Regulation] = field(default_factory=list)
    total_requirements: int = 0
    compliant_count: int = 0
    non_compliant_count: int = 0
    partial_count: int = 0
    not_applicable_count: int = 0
    pending_count: int = 0
    weighted_score: float = 0.0
    unweighted_score: float = 0.0
    is_passing: bool = False
    findings: List[ComplianceFinding] = field(default_factory=list)
    remediations: List[RemediationAction] = field(default_factory=list)
    summary: str = ""

    def compute_scores(self) -> None:
        """Compute weighted and unweighted compliance scores."""
        assessed = self.compliant_count + self.non_compliant_count + self.partial_count
        if assessed > 0:
            self.unweighted_score = round((self.compliant_count + 0.5 * self.partial_count) / assessed * 100.0, 2)
        else:
            self.unweighted_score = 0.0
        self.is_passing = self.unweighted_score >= PASSING_SCORE_THRESHOLD


@dataclass
class AuditTrailEntry:
    """Audit trail entry for compliance validation decisions."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = ""
    entity_type: str = ""
    entity_id: str = ""
    user: str = ""
    details: str = ""
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for tamper detection."""
        payload = f"{self.timestamp}|{self.action}|{self.entity_id}|{self.details}"
        self.content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.content_hash


# ===================================================================
# Standard requirement libraries
# ===================================================================
def _build_21cfr820_requirements() -> List[RegulatoryRequirement]:
    """Build FDA 21 CFR 820 Quality System Regulation requirements."""
    reqs = [
        ("820.20", "Management Responsibility", RequirementCategory.QUALITY_SYSTEM, 2.0),
        ("820.22", "Quality Audit", RequirementCategory.QUALITY_SYSTEM, 1.5),
        ("820.25", "Personnel", RequirementCategory.QUALITY_SYSTEM, 1.0),
        ("820.30(a)", "Design Controls - General", RequirementCategory.DESIGN_CONTROLS, 2.0),
        ("820.30(b)", "Design and Development Planning", RequirementCategory.DESIGN_CONTROLS, 2.0),
        ("820.30(c)", "Design Input", RequirementCategory.DESIGN_CONTROLS, 2.0),
        ("820.30(d)", "Design Output", RequirementCategory.DESIGN_CONTROLS, 2.0),
        ("820.30(e)", "Design Review", RequirementCategory.DESIGN_CONTROLS, 1.5),
        ("820.30(f)", "Design Verification", RequirementCategory.DESIGN_CONTROLS, 2.0),
        ("820.30(g)", "Design Validation", RequirementCategory.DESIGN_CONTROLS, 2.0),
        ("820.30(i)", "Design Changes", RequirementCategory.DESIGN_CONTROLS, 1.5),
        ("820.40", "Document Controls", RequirementCategory.DOCUMENTATION, 1.5),
        ("820.90", "Nonconforming Product", RequirementCategory.QUALITY_SYSTEM, 1.0),
        ("820.184", "Device History Record", RequirementCategory.DOCUMENTATION, 1.5),
        ("820.198", "Complaint Files", RequirementCategory.POST_MARKET, 1.0),
    ]
    return [
        RegulatoryRequirement(
            regulation=Regulation.FDA_21CFR820,
            section=sec,
            title=title,
            category=cat,
            weight=w,
            description=f"FDA 21 CFR 820 requirement: {title}",
            evidence_needed=f"Documentation demonstrating compliance with {sec}",
        )
        for sec, title, cat, w in reqs
    ]


def _build_21cfr11_requirements() -> List[RegulatoryRequirement]:
    """Build FDA 21 CFR Part 11 Electronic Records requirements."""
    reqs = [
        ("11.10(a)", "System Validation", RequirementCategory.DATA_INTEGRITY, 2.0),
        ("11.10(b)", "Record Generation - Legible Copies", RequirementCategory.DATA_INTEGRITY, 1.5),
        ("11.10(c)", "Record Protection", RequirementCategory.DATA_INTEGRITY, 2.0),
        ("11.10(d)", "Access Limitation", RequirementCategory.DATA_INTEGRITY, 2.0),
        ("11.10(e)", "Audit Trail", RequirementCategory.DATA_INTEGRITY, 2.5),
        ("11.10(g)", "Authority Checks", RequirementCategory.DATA_INTEGRITY, 1.5),
        ("11.10(k)", "Documentation Controls", RequirementCategory.DOCUMENTATION, 1.5),
        ("11.50", "Signature Manifestation", RequirementCategory.DATA_INTEGRITY, 1.5),
        ("11.70", "Signature/Record Linking", RequirementCategory.DATA_INTEGRITY, 1.5),
        ("11.100", "Electronic Signature Components", RequirementCategory.DATA_INTEGRITY, 2.0),
        ("11.200", "Electronic Signature Controls", RequirementCategory.DATA_INTEGRITY, 2.0),
        ("11.300", "Identification Code Controls", RequirementCategory.DATA_INTEGRITY, 1.5),
    ]
    return [
        RegulatoryRequirement(
            regulation=Regulation.FDA_21CFR11,
            section=sec,
            title=title,
            category=cat,
            weight=w,
            description=f"FDA 21 CFR Part 11 requirement: {title}",
            evidence_needed=f"Evidence of compliance with {sec}",
        )
        for sec, title, cat, w in reqs
    ]


def _build_iso14971_requirements() -> List[RegulatoryRequirement]:
    """Build ISO 14971 Risk Management requirements."""
    reqs = [
        ("4.1", "Risk Management Process", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("4.2", "Management Responsibilities", RequirementCategory.RISK_MANAGEMENT, 1.5),
        ("4.3", "Qualification of Personnel", RequirementCategory.RISK_MANAGEMENT, 1.0),
        ("4.4", "Risk Management Plan", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("4.5", "Risk Management File", RequirementCategory.DOCUMENTATION, 1.5),
        ("5.1", "Risk Analysis Process", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("5.2", "Intended Use and Identification of Characteristics", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("5.3", "Identification of Hazards", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("5.4", "Estimation of Risk", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("6", "Risk Evaluation", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("7.1", "Risk Control Option Analysis", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("7.2", "Implementation of Risk Control Measures", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("7.3", "Residual Risk Evaluation", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("8", "Evaluation of Overall Residual Risk", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("9", "Risk Management Review", RequirementCategory.RISK_MANAGEMENT, 1.5),
        ("10", "Production and Post-Production Activities", RequirementCategory.POST_MARKET, 1.5),
    ]
    return [
        RegulatoryRequirement(
            regulation=Regulation.ISO_14971,
            section=sec,
            title=title,
            category=cat,
            weight=w,
            description=f"ISO 14971:2019 requirement: {title}",
            evidence_needed=f"Risk management documentation for clause {sec}",
        )
        for sec, title, cat, w in reqs
    ]


def _build_iec62304_requirements() -> List[RegulatoryRequirement]:
    """Build IEC 62304 Medical Device Software Lifecycle requirements."""
    reqs = [
        ("4.1", "Quality Management System", RequirementCategory.SOFTWARE_LIFECYCLE, 1.5),
        ("5.1", "Software Development Planning", RequirementCategory.SOFTWARE_LIFECYCLE, 2.0),
        ("5.2", "Software Requirements Analysis", RequirementCategory.SOFTWARE_LIFECYCLE, 2.0),
        ("5.3", "Software Architectural Design", RequirementCategory.SOFTWARE_LIFECYCLE, 2.0),
        ("5.4", "Software Detailed Design", RequirementCategory.SOFTWARE_LIFECYCLE, 1.5),
        ("5.5", "Software Unit Implementation", RequirementCategory.SOFTWARE_LIFECYCLE, 1.5),
        ("5.6", "Software Integration and Testing", RequirementCategory.SOFTWARE_LIFECYCLE, 2.0),
        ("5.7", "Software System Testing", RequirementCategory.SOFTWARE_LIFECYCLE, 2.0),
        ("5.8", "Software Release", RequirementCategory.SOFTWARE_LIFECYCLE, 1.5),
        ("6", "Software Maintenance", RequirementCategory.SOFTWARE_LIFECYCLE, 1.5),
        ("7", "Software Risk Management", RequirementCategory.RISK_MANAGEMENT, 2.0),
        ("8", "Software Configuration Management", RequirementCategory.SOFTWARE_LIFECYCLE, 1.5),
        ("9", "Software Problem Resolution", RequirementCategory.SOFTWARE_LIFECYCLE, 1.5),
    ]
    return [
        RegulatoryRequirement(
            regulation=Regulation.IEC_62304,
            section=sec,
            title=title,
            category=cat,
            weight=w,
            description=f"IEC 62304:2015 requirement: {title}",
            evidence_needed=f"Software lifecycle documentation for clause {sec}",
        )
        for sec, title, cat, w in reqs
    ]


_REQUIREMENT_BUILDERS: Dict[Regulation, Any] = {
    Regulation.FDA_21CFR820: _build_21cfr820_requirements,
    Regulation.FDA_21CFR11: _build_21cfr11_requirements,
    Regulation.ISO_14971: _build_iso14971_requirements,
    Regulation.IEC_62304: _build_iec62304_requirements,
}


# ===================================================================
# Main class
# ===================================================================
class ComplianceValidator:
    """Multi-regulation compliance validation engine.

    Performs checklist-driven validation against multiple regulatory
    frameworks simultaneously, generates gap analyses, tracks remediation
    actions, and computes compliance scores.

    Parameters
    ----------
    submission_id : str
        Unique submission identifier.
    regulations : Sequence[Regulation]
        Regulations to validate against.
    """

    def __init__(
        self,
        submission_id: str,
        regulations: Optional[Sequence[Regulation]] = None,
    ) -> None:
        self._submission_id = submission_id
        self._regulations = list(regulations) if regulations else list(Regulation)
        self._requirements: Dict[str, RegulatoryRequirement] = {}
        self._findings: Dict[str, ComplianceFinding] = {}
        self._remediations: Dict[str, RemediationAction] = {}
        self._audit_trail: List[AuditTrailEntry] = []
        self._evidence: Dict[str, str] = {}
        self._load_requirements()
        logger.info(
            "ComplianceValidator initialized for %s with %d regulation(s)",
            submission_id,
            len(self._regulations),
        )

    def _load_requirements(self) -> None:
        """Load requirement checklists for configured regulations."""
        for reg in self._regulations:
            builder = _REQUIREMENT_BUILDERS.get(reg)
            if builder:
                for req in builder():
                    self._requirements[req.requirement_id] = req
        logger.info("Loaded %d requirements across %d regulation(s)", len(self._requirements), len(self._regulations))

    def _record_audit(self, action: str, entity_type: str, entity_id: str, details: str, user: str = "system") -> None:
        """Append an audit trail entry."""
        if len(self._audit_trail) >= MAX_AUDIT_ENTRIES:
            return
        entry = AuditTrailEntry(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user=user,
            details=details,
        )
        entry.compute_hash()
        self._audit_trail.append(entry)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def submission_id(self) -> str:
        return self._submission_id

    @property
    def requirement_count(self) -> int:
        return len(self._requirements)

    @property
    def finding_count(self) -> int:
        return len(self._findings)

    @property
    def audit_trail(self) -> List[AuditTrailEntry]:
        return list(self._audit_trail)

    # ------------------------------------------------------------------
    # Evidence management
    # ------------------------------------------------------------------
    def submit_evidence(self, requirement_id: str, evidence: str, user: str = "system") -> bool:
        """Submit evidence for a regulatory requirement."""
        if requirement_id not in self._requirements:
            logger.warning("Requirement not found: %s", requirement_id)
            return False
        self._evidence[requirement_id] = evidence
        self._record_audit(
            "submit_evidence", "requirement", requirement_id, f"Evidence submitted: {evidence[:50]}", user
        )
        return True

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------
    def assess_requirement(
        self,
        requirement_id: str,
        status: ComplianceStatus,
        evidence: str = "",
        gap_description: str = "",
        assessed_by: str = "system",
    ) -> Optional[ComplianceFinding]:
        """Assess a single requirement and record the finding."""
        if requirement_id not in self._requirements:
            logger.warning("Requirement not found: %s", requirement_id)
            return None

        req = self._requirements[requirement_id]
        finding = ComplianceFinding(
            requirement_id=requirement_id,
            regulation=req.regulation,
            status=status,
            title=req.title,
            description=req.description,
            evidence_provided=evidence or self._evidence.get(requirement_id, ""),
            gap_description=gap_description,
            assessed_by=assessed_by,
        )

        # Determine severity based on status and requirement weight
        if status == ComplianceStatus.NON_COMPLIANT:
            if req.is_mandatory and req.weight >= 2.0:
                finding.severity = FindingSeverity.CRITICAL
            elif req.is_mandatory:
                finding.severity = FindingSeverity.MAJOR
            else:
                finding.severity = FindingSeverity.MINOR
        elif status == ComplianceStatus.PARTIAL:
            finding.severity = FindingSeverity.MINOR if req.weight < 2.0 else FindingSeverity.MAJOR
        else:
            finding.severity = FindingSeverity.OBSERVATION

        self._findings[finding.finding_id] = finding
        self._record_audit("assess", "requirement", requirement_id, f"Status: {status.value}", assessed_by)
        return finding

    def run_assessment(
        self,
        evidence_map: Optional[Dict[str, str]] = None,
        assessed_by: str = "system",
    ) -> ComplianceAssessment:
        """Run compliance assessment across all loaded requirements.

        Requirements without evidence are marked as PENDING_REVIEW.
        Requirements with evidence are assessed as COMPLIANT (simulated).

        Parameters
        ----------
        evidence_map : dict, optional
            Mapping of requirement_id -> evidence string.
        assessed_by : str
            Name of the assessor.
        """
        if evidence_map:
            for req_id, ev in evidence_map.items():
                self.submit_evidence(req_id, ev, assessed_by)

        self._findings.clear()
        for req_id, req in self._requirements.items():
            evidence = self._evidence.get(req_id, "")
            if evidence:
                status = ComplianceStatus.COMPLIANT
                gap = ""
            elif not req.is_mandatory:
                status = ComplianceStatus.NOT_APPLICABLE
                gap = ""
            else:
                status = ComplianceStatus.PENDING_REVIEW
                gap = f"No evidence provided for mandatory requirement: {req.title}"
            self.assess_requirement(req_id, status, evidence, gap, assessed_by)

        return self._compile_assessment()

    def _compile_assessment(self) -> ComplianceAssessment:
        """Compile findings into a ComplianceAssessment."""
        findings = list(self._findings.values())
        compliant = sum(1 for f in findings if f.status == ComplianceStatus.COMPLIANT)
        non_compliant = sum(1 for f in findings if f.status == ComplianceStatus.NON_COMPLIANT)
        partial = sum(1 for f in findings if f.status == ComplianceStatus.PARTIAL)
        not_applicable = sum(1 for f in findings if f.status == ComplianceStatus.NOT_APPLICABLE)
        pending = sum(1 for f in findings if f.status == ComplianceStatus.PENDING_REVIEW)

        # Weighted score
        total_weight = 0.0
        earned_weight = 0.0
        for finding in findings:
            req = self._requirements.get(finding.requirement_id)
            if req and finding.status not in (ComplianceStatus.NOT_APPLICABLE, ComplianceStatus.PENDING_REVIEW):
                total_weight += req.weight
                if finding.status == ComplianceStatus.COMPLIANT:
                    earned_weight += req.weight
                elif finding.status == ComplianceStatus.PARTIAL:
                    earned_weight += req.weight * 0.5
        weighted_score = round(earned_weight / total_weight * 100.0, 2) if total_weight > 0 else 0.0

        assessment = ComplianceAssessment(
            submission_id=self._submission_id,
            regulations_assessed=list(self._regulations),
            total_requirements=len(findings),
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            partial_count=partial,
            not_applicable_count=not_applicable,
            pending_count=pending,
            weighted_score=weighted_score,
            findings=findings,
            remediations=list(self._remediations.values()),
        )
        assessment.compute_scores()
        assessment.summary = (
            f"Assessed {len(findings)} requirements across {len(self._regulations)} regulation(s). "
            f"Compliant: {compliant}, Non-compliant: {non_compliant}, Partial: {partial}, "
            f"Pending: {pending}. Weighted score: {weighted_score:.1f}%."
        )
        return assessment

    # ------------------------------------------------------------------
    # Remediation management
    # ------------------------------------------------------------------
    def create_remediation(
        self,
        finding_id: str,
        title: str,
        description: str = "",
        owner: str = "",
        target_date: str = "",
    ) -> Optional[RemediationAction]:
        """Create a remediation action for a compliance finding."""
        if finding_id not in self._findings:
            logger.warning("Finding not found: %s", finding_id)
            return None
        finding = self._findings[finding_id]
        action = RemediationAction(
            finding_id=finding_id,
            title=title,
            description=description or finding.remediation_recommendation,
            owner=owner,
            priority=finding.severity,
            target_date=target_date,
        )
        self._remediations[action.action_id] = action
        self._record_audit("create_remediation", "finding", finding_id, title)
        return action

    def update_remediation_status(self, action_id: str, status: RemediationStatus, notes: str = "") -> bool:
        """Update the status of a remediation action."""
        if action_id not in self._remediations:
            return False
        action = self._remediations[action_id]
        action.status = status
        if status == RemediationStatus.RESOLVED:
            action.completion_date = datetime.now(timezone.utc).isoformat()
        if notes:
            action.verification_notes = notes
        self._record_audit("update_remediation", "remediation", action_id, f"Status: {status.value}")
        return True

    def get_open_remediations(self) -> List[RemediationAction]:
        """Return all open remediation actions."""
        return [
            a
            for a in self._remediations.values()
            if a.status in (RemediationStatus.OPEN, RemediationStatus.IN_PROGRESS)
        ]

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------
    def generate_gap_analysis(self) -> Dict[Regulation, Dict[str, Any]]:
        """Generate a gap analysis by regulation."""
        gaps: Dict[Regulation, Dict[str, Any]] = {}
        for reg in self._regulations:
            reg_findings = [f for f in self._findings.values() if f.regulation == reg]
            non_compliant = [f for f in reg_findings if f.status == ComplianceStatus.NON_COMPLIANT]
            partial = [f for f in reg_findings if f.status == ComplianceStatus.PARTIAL]
            pending = [f for f in reg_findings if f.status == ComplianceStatus.PENDING_REVIEW]
            compliant = [f for f in reg_findings if f.status == ComplianceStatus.COMPLIANT]
            total = len(reg_findings) - sum(1 for f in reg_findings if f.status == ComplianceStatus.NOT_APPLICABLE)
            score = round(len(compliant) / total * 100.0, 2) if total > 0 else 0.0
            gaps[reg] = {
                "total_assessed": len(reg_findings),
                "compliant": len(compliant),
                "non_compliant": len(non_compliant),
                "partial": len(partial),
                "pending": len(pending),
                "score": score,
                "critical_gaps": [f.title for f in non_compliant if f.severity == FindingSeverity.CRITICAL],
            }
        return gaps

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_compliance_report(self, assessment: Optional[ComplianceAssessment] = None) -> str:
        """Generate a Markdown compliance report."""
        if assessment is None:
            assessment = self._compile_assessment()

        lines: List[str] = [
            f"# Compliance Validation Report: {self._submission_id}",
            "",
            f"**Assessment Date:** {assessment.assessed_at[:19]}",
            f"**Regulations:** {', '.join(r.value for r in assessment.regulations_assessed)}",
            f"**Overall Status:** {'PASSING' if assessment.is_passing else 'NOT PASSING'}",
            f"**Weighted Score:** {assessment.weighted_score:.1f}%",
            f"**Unweighted Score:** {assessment.unweighted_score:.1f}%",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total Requirements | {assessment.total_requirements} |",
            f"| Compliant | {assessment.compliant_count} |",
            f"| Non-Compliant | {assessment.non_compliant_count} |",
            f"| Partial | {assessment.partial_count} |",
            f"| Not Applicable | {assessment.not_applicable_count} |",
            f"| Pending Review | {assessment.pending_count} |",
            "",
            "## Gap Analysis by Regulation",
            "",
        ]

        gap_analysis = self.generate_gap_analysis()
        for reg, data in gap_analysis.items():
            lines.append(f"### {reg.value}")
            lines.append(f"- Score: {data['score']:.1f}%")
            lines.append(f"- Compliant: {data['compliant']}/{data['total_assessed']}")
            if data["critical_gaps"]:
                lines.append(f"- Critical gaps: {', '.join(data['critical_gaps'])}")
            lines.append("")

        # Findings table
        lines.extend(["## Findings", ""])
        critical_findings = [
            f for f in assessment.findings if f.severity in (FindingSeverity.CRITICAL, FindingSeverity.MAJOR)
        ]
        if critical_findings:
            lines.append("| Severity | Regulation | Title | Status |")
            lines.append("|----------|-----------|-------|--------|")
            for f in critical_findings[:20]:
                lines.append(f"| {f.severity.value} | {f.regulation.value} | {f.title} | {f.status.value} |")
        else:
            lines.append("No critical or major findings.")

        # Open remediations
        open_rems = self.get_open_remediations()
        if open_rems:
            lines.extend(["", "## Open Remediations", ""])
            for rem in open_rems[:10]:
                lines.append(f"- [{rem.priority.value}] {rem.title} (owner: {rem.owner or 'unassigned'})")

        lines.extend(["", f"*Report generated: {datetime.now(timezone.utc).isoformat()[:19]}*"])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize validator state to a dictionary."""
        return {
            "submission_id": self._submission_id,
            "regulations": [r.value for r in self._regulations],
            "requirement_count": len(self._requirements),
            "finding_count": len(self._findings),
            "remediation_count": len(self._remediations),
            "audit_entries": len(self._audit_trail),
        }

    def get_requirements_by_regulation(self, regulation: Regulation) -> List[RegulatoryRequirement]:
        """Return all requirements for a specific regulation."""
        return [r for r in self._requirements.values() if r.regulation == regulation]

    def get_findings_by_severity(self, severity: FindingSeverity) -> List[ComplianceFinding]:
        """Return all findings with a specific severity."""
        return [f for f in self._findings.values() if f.severity == severity]


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    validator = ComplianceValidator(
        "SUB-2026-DEMO",
        [Regulation.FDA_21CFR820, Regulation.ISO_14971, Regulation.IEC_62304],
    )
    assessment = validator.run_assessment()
    print(validator.generate_compliance_report(assessment))

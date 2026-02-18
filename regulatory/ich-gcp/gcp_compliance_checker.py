"""ICH E6(R3) Good Clinical Practice compliance checker.

Assesses clinical trial compliance against ICH E6(R3) principles
published September 2025, with structured findings, severity
classification, and compliance scoring.

References:
    - ICH E6(R3): Good Clinical Practice — published September 2025
    - ICH E8(R1): General Considerations for Clinical Studies
    - ICH E9(R1): Statistical Principles — Estimands
    - 21 CFR Parts 11, 50, 56, 312: FDA clinical trial regulations
    - EU Clinical Trials Regulation (EU) 2014/536

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GCPPrinciple(str, Enum):
    """ICH E6(R3) GCP fundamental principles.

    These principles form the foundation of GCP compliance
    as defined in the September 2025 publication.
    """

    ETHICAL_CONDUCT = "ethical_conduct"
    SCIENTIFIC_SOUNDNESS = "scientific_soundness"
    PARTICIPANT_RIGHTS = "participant_rights"
    INFORMED_CONSENT = "informed_consent"
    IRB_IEC_REVIEW = "irb_iec_review"
    QUALIFIED_PERSONNEL = "qualified_personnel"
    DATA_INTEGRITY = "data_integrity"
    PROTOCOL_COMPLIANCE = "protocol_compliance"
    INVESTIGATIONAL_PRODUCT = "investigational_product"
    QUALITY_MANAGEMENT = "quality_management"
    RECORD_KEEPING = "record_keeping"
    PARTICIPANT_SAFETY = "participant_safety"
    RISK_PROPORTIONATE = "risk_proportionate_approach"


class ComplianceLevel(str, Enum):
    """Compliance assessment levels for findings."""

    COMPLIANT = "compliant"
    MINOR_FINDING = "minor_finding"
    MAJOR_FINDING = "major_finding"
    CRITICAL_FINDING = "critical_finding"
    NOT_ASSESSED = "not_assessed"


class FindingCategory(str, Enum):
    """Categories of compliance findings."""

    DOCUMENTATION = "documentation"
    PROCESS = "process"
    TRAINING = "training"
    OVERSIGHT = "oversight"
    DATA_MANAGEMENT = "data_management"
    SAFETY_REPORTING = "safety_reporting"
    CONSENT = "consent"
    REGULATORY = "regulatory"
    QUALITY = "quality"
    TECHNOLOGY = "technology"


class TrialPhase(str, Enum):
    """Clinical trial phases."""

    PHASE_I = "phase_i"
    PHASE_II = "phase_ii"
    PHASE_III = "phase_iii"
    PHASE_IV = "phase_iv"
    OBSERVATIONAL = "observational"
    REGISTRY = "registry"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ComplianceFinding:
    """A single compliance finding from an assessment.

    Attributes:
        finding_id: Unique finding identifier.
        principle: GCP principle assessed.
        level: Compliance level determined.
        category: Finding category.
        description: Detailed finding description.
        reference: Regulatory reference (ICH section, CFR cite).
        remediation: Recommended remediation action.
        evidence: Supporting evidence or observation.
        assessed_at: When the finding was assessed.
    """

    finding_id: str
    principle: GCPPrinciple
    level: ComplianceLevel
    category: FindingCategory
    description: str = ""
    reference: str = ""
    remediation: str = ""
    evidence: str = ""
    assessed_at: str = ""

    def __post_init__(self) -> None:
        if not self.assessed_at:
            self.assessed_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AssessmentArea:
    """A specific area being assessed for GCP compliance.

    Attributes:
        area_id: Unique area identifier.
        name: Area name.
        principle: Primary GCP principle.
        checks: Individual compliance checks within the area.
        overall_level: Overall compliance level for the area.
    """

    area_id: str
    name: str
    principle: GCPPrinciple
    checks: list[dict[str, Any]] = field(default_factory=list)
    overall_level: ComplianceLevel = ComplianceLevel.NOT_ASSESSED


@dataclass
class ComplianceReport:
    """Complete GCP compliance assessment report.

    Attributes:
        report_id: Unique report identifier.
        trial_id: Identifier of the trial assessed.
        trial_phase: Phase of the trial.
        assessor: Person or system performing the assessment.
        assessment_date: Date of the assessment.
        scope: Scope of the assessment.
        findings: All compliance findings.
        areas_assessed: Assessment areas with results.
        overall_score: Compliance score (0.0-1.0).
        summary: Executive summary.
        recommendations: Prioritized recommendations.
    """

    report_id: str
    trial_id: str = ""
    trial_phase: TrialPhase = TrialPhase.PHASE_III
    assessor: str = ""
    assessment_date: str = ""
    scope: str = ""
    findings: list[ComplianceFinding] = field(default_factory=list)
    areas_assessed: list[AssessmentArea] = field(default_factory=list)
    overall_score: float = 0.0
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Standard assessment checklist
# ---------------------------------------------------------------------------

_GCP_CHECKLIST: list[dict[str, Any]] = [
    {
        "principle": GCPPrinciple.ETHICAL_CONDUCT,
        "area": "Ethical Foundation",
        "checks": [
            "Trial conducted in accordance with Declaration of Helsinki principles",
            "Benefits justify risks to participants",
            "Participant rights and safety prioritized over scientific interests",
            "Vulnerable populations have additional protections",
        ],
    },
    {
        "principle": GCPPrinciple.SCIENTIFIC_SOUNDNESS,
        "area": "Scientific Validity",
        "checks": [
            "Protocol based on adequate nonclinical and clinical information",
            "Statistical design adequate for objectives (ICH E9(R1) estimands)",
            "Endpoints clinically meaningful and properly defined",
            "Sample size justified with power calculations",
        ],
    },
    {
        "principle": GCPPrinciple.INFORMED_CONSENT,
        "area": "Informed Consent Process",
        "checks": [
            "Consent obtained before any trial-related procedures",
            "Consent document IRB/IEC approved and current version used",
            "AI/ML components disclosed in consent (per E6(R3) Section 4.8)",
            "Participants informed of data use for model training",
            "Right to withdraw clearly communicated",
            "Consent process documented in source records",
        ],
    },
    {
        "principle": GCPPrinciple.IRB_IEC_REVIEW,
        "area": "IRB/IEC Oversight",
        "checks": [
            "IRB/IEC approval obtained before trial initiation",
            "Continuing review conducted at appropriate intervals",
            "Amendments submitted and approved before implementation",
            "Safety reports submitted to IRB/IEC per regulations",
            "AI/ML modifications reviewed by IRB/IEC when applicable",
        ],
    },
    {
        "principle": GCPPrinciple.QUALIFIED_PERSONNEL,
        "area": "Personnel Qualifications",
        "checks": [
            "PI qualified by education, training, and experience",
            "Sub-investigators appropriately delegated and trained",
            "Data scientists/ML engineers have appropriate GCP training",
            "Delegation log current and complete",
        ],
    },
    {
        "principle": GCPPrinciple.DATA_INTEGRITY,
        "area": "Data Integrity and Management",
        "checks": [
            "Data recorded in accordance with protocol and source documents",
            "ALCOA+ principles (Attributable, Legible, Contemporaneous, Original, Accurate) followed",
            "Electronic systems validated per 21 CFR Part 11",
            "Audit trail maintained for all data changes",
            "AI/ML model inputs and outputs traceable to source data",
            "Federated learning data provenance documented",
        ],
    },
    {
        "principle": GCPPrinciple.PROTOCOL_COMPLIANCE,
        "area": "Protocol Adherence",
        "checks": [
            "Protocol deviations identified and documented",
            "Protocol amendments properly submitted and approved",
            "Eligibility criteria consistently applied",
            "Visit schedules and assessments per protocol",
        ],
    },
    {
        "principle": GCPPrinciple.PARTICIPANT_SAFETY,
        "area": "Safety Monitoring",
        "checks": [
            "Adverse events identified, documented, and reported per timeline",
            "SAEs reported within 24 hours per regulatory requirements",
            "DSMB/DMC in place for appropriate trials",
            "AI/ML safety boundaries defined and monitored",
            "Emergency unblinding procedures available",
        ],
    },
    {
        "principle": GCPPrinciple.QUALITY_MANAGEMENT,
        "area": "Quality Management System",
        "checks": [
            "Risk-proportionate quality management approach per E6(R3)",
            "Critical processes and data identified",
            "Risk assessment performed and documented",
            "Centralized monitoring implemented where appropriate",
            "CAPAs documented and tracked to resolution",
        ],
    },
    {
        "principle": GCPPrinciple.RECORD_KEEPING,
        "area": "Essential Documents and Records",
        "checks": [
            "Trial master file maintained and current",
            "Essential documents per E6(R3) Section 8 maintained",
            "Records retained for required period (per ICH E6(R3))",
            "Electronic records comply with 21 CFR Part 11",
        ],
    },
    {
        "principle": GCPPrinciple.RISK_PROPORTIONATE,
        "area": "Risk-Proportionate Approach (E6(R3) Innovation)",
        "checks": [
            "Risk-based monitoring plan implemented",
            "Technology-enabled approaches validated (eConsent, ePRO)",
            "AI/ML risk proportionate to intended use and impact",
            "Decentralized trial elements appropriately governed",
        ],
    },
]


# ---------------------------------------------------------------------------
# GCP Compliance Checker
# ---------------------------------------------------------------------------


class GCPComplianceChecker:
    """ICH E6(R3) GCP compliance assessment engine.

    Provides structured compliance checks against ICH E6(R3)
    principles, generates findings with severity classification,
    and produces compliance scores excluding NOT_ASSESSED items
    from the scoring denominator.

    Reference: ICH E6(R3) published September 2025.
    """

    def __init__(self) -> None:
        self._reports: dict[str, ComplianceReport] = {}
        self._checklist = list(_GCP_CHECKLIST)

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess_trial(
        self,
        trial_id: str,
        trial_phase: TrialPhase | str = TrialPhase.PHASE_III,
        assessor: str = "",
        scope: str = "Full GCP compliance assessment per ICH E6(R3)",
        check_results: dict[str, ComplianceLevel] | None = None,
    ) -> ComplianceReport:
        """Perform a GCP compliance assessment.

        Args:
            trial_id: Identifier of the trial to assess.
            trial_phase: Phase of the trial.
            assessor: Person or system performing the assessment.
            scope: Scope description for the assessment.
            check_results: Optional pre-determined results keyed by check text.
                Checks not in this dict are marked NOT_ASSESSED.

        Returns:
            A ComplianceReport with all findings and overall score.
        """
        if isinstance(trial_phase, str):
            trial_phase = TrialPhase(trial_phase)

        report_id = f"GCP-{trial_id}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        report = ComplianceReport(
            report_id=report_id,
            trial_id=trial_id,
            trial_phase=trial_phase,
            assessor=assessor,
            scope=scope,
        )

        results = check_results or {}
        finding_counter = 0

        for area_def in self._checklist:
            principle = area_def["principle"]
            area_name = area_def["area"]
            checks = area_def["checks"]

            area = AssessmentArea(
                area_id=f"{report_id}-{principle.value}",
                name=area_name,
                principle=principle,
            )

            area_levels: list[ComplianceLevel] = []

            for check_text in checks:
                level = results.get(check_text, ComplianceLevel.NOT_ASSESSED)
                area.checks.append(
                    {
                        "check": check_text,
                        "level": level.value,
                    }
                )
                area_levels.append(level)

                # Generate findings for non-compliant checks
                if level in (
                    ComplianceLevel.MINOR_FINDING,
                    ComplianceLevel.MAJOR_FINDING,
                    ComplianceLevel.CRITICAL_FINDING,
                ):
                    finding_counter += 1
                    finding = ComplianceFinding(
                        finding_id=f"{report_id}-F{finding_counter:03d}",
                        principle=principle,
                        level=level,
                        category=self._categorize_finding(principle),
                        description=f"Non-compliance identified: {check_text}",
                        reference=f"ICH E6(R3) — {principle.value}",
                        remediation=self._suggest_remediation(check_text, level),
                    )
                    report.findings.append(finding)

            # Determine area-level compliance (worst finding wins)
            assessed_levels = [lv for lv in area_levels if lv != ComplianceLevel.NOT_ASSESSED]
            if assessed_levels:
                severity_order = [
                    ComplianceLevel.COMPLIANT,
                    ComplianceLevel.MINOR_FINDING,
                    ComplianceLevel.MAJOR_FINDING,
                    ComplianceLevel.CRITICAL_FINDING,
                ]
                area.overall_level = max(assessed_levels, key=lambda lv: severity_order.index(lv))
            else:
                area.overall_level = ComplianceLevel.NOT_ASSESSED

            report.areas_assessed.append(area)

        # Calculate overall compliance score
        report.overall_score = self._calculate_score(report)
        report.summary = self._generate_summary(report)
        report.recommendations = self._generate_recommendations(report)

        self._reports[report_id] = report
        logger.info(
            "GCP assessment complete for trial '%s': score=%.1f%%, findings=%d",
            trial_id,
            report.overall_score * 100,
            len(report.findings),
        )
        return report

    def add_finding(
        self,
        report_id: str,
        principle: GCPPrinciple | str,
        level: ComplianceLevel | str,
        category: FindingCategory | str,
        description: str,
        reference: str = "",
        remediation: str = "",
    ) -> str | None:
        """Add a manual finding to an existing report.

        Returns:
            The finding ID, or None if the report was not found.
        """
        report = self._reports.get(report_id)
        if report is None:
            return None
        if isinstance(principle, str):
            principle = GCPPrinciple(principle)
        if isinstance(level, str):
            level = ComplianceLevel(level)
        if isinstance(category, str):
            category = FindingCategory(category)

        finding_id = f"{report_id}-F{len(report.findings) + 1:03d}"
        finding = ComplianceFinding(
            finding_id=finding_id,
            principle=principle,
            level=level,
            category=category,
            description=description,
            reference=reference or f"ICH E6(R3) — {principle.value}",
            remediation=remediation,
        )
        report.findings.append(finding)
        report.overall_score = self._calculate_score(report)
        return finding_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_report(self, report_id: str) -> ComplianceReport | None:
        """Retrieve a compliance report."""
        return self._reports.get(report_id)

    def get_findings_by_level(
        self,
        report_id: str,
        level: ComplianceLevel,
    ) -> list[ComplianceFinding]:
        """Get findings filtered by compliance level."""
        report = self._reports.get(report_id)
        if report is None:
            return []
        return [f for f in report.findings if f.level == level]

    def get_checklist(self) -> list[dict[str, Any]]:
        """Return the GCP compliance checklist."""
        return list(self._checklist)

    # ------------------------------------------------------------------
    # Score calculation
    # ------------------------------------------------------------------

    def _calculate_score(self, report: ComplianceReport) -> float:
        """Calculate the overall compliance score.

        Excludes NOT_ASSESSED findings from the denominator so that
        partial assessments still produce meaningful scores.

        Scoring:
            - COMPLIANT = 1.0
            - MINOR_FINDING = 0.7
            - MAJOR_FINDING = 0.3
            - CRITICAL_FINDING = 0.0
            - NOT_ASSESSED = excluded from calculation
        """
        level_scores: dict[ComplianceLevel, float] = {
            ComplianceLevel.COMPLIANT: 1.0,
            ComplianceLevel.MINOR_FINDING: 0.7,
            ComplianceLevel.MAJOR_FINDING: 0.3,
            ComplianceLevel.CRITICAL_FINDING: 0.0,
        }

        total_score = 0.0
        assessed_count = 0

        for area in report.areas_assessed:
            for check in area.checks:
                level = ComplianceLevel(check["level"])
                if level == ComplianceLevel.NOT_ASSESSED:
                    continue  # Exclude from denominator
                total_score += level_scores.get(level, 0.0)
                assessed_count += 1

        if assessed_count == 0:
            return 0.0

        return total_score / assessed_count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize_finding(principle: GCPPrinciple) -> FindingCategory:
        """Map a GCP principle to a finding category."""
        mapping: dict[GCPPrinciple, FindingCategory] = {
            GCPPrinciple.ETHICAL_CONDUCT: FindingCategory.REGULATORY,
            GCPPrinciple.SCIENTIFIC_SOUNDNESS: FindingCategory.PROCESS,
            GCPPrinciple.PARTICIPANT_RIGHTS: FindingCategory.CONSENT,
            GCPPrinciple.INFORMED_CONSENT: FindingCategory.CONSENT,
            GCPPrinciple.IRB_IEC_REVIEW: FindingCategory.OVERSIGHT,
            GCPPrinciple.QUALIFIED_PERSONNEL: FindingCategory.TRAINING,
            GCPPrinciple.DATA_INTEGRITY: FindingCategory.DATA_MANAGEMENT,
            GCPPrinciple.PROTOCOL_COMPLIANCE: FindingCategory.PROCESS,
            GCPPrinciple.INVESTIGATIONAL_PRODUCT: FindingCategory.PROCESS,
            GCPPrinciple.QUALITY_MANAGEMENT: FindingCategory.QUALITY,
            GCPPrinciple.RECORD_KEEPING: FindingCategory.DOCUMENTATION,
            GCPPrinciple.PARTICIPANT_SAFETY: FindingCategory.SAFETY_REPORTING,
            GCPPrinciple.RISK_PROPORTIONATE: FindingCategory.QUALITY,
        }
        return mapping.get(principle, FindingCategory.PROCESS)

    @staticmethod
    def _suggest_remediation(check_text: str, level: ComplianceLevel) -> str:
        """Generate a remediation recommendation based on finding severity."""
        urgency = {
            ComplianceLevel.MINOR_FINDING: "Address within 30 days",
            ComplianceLevel.MAJOR_FINDING: "Address within 7 days",
            ComplianceLevel.CRITICAL_FINDING: "Immediate corrective action required",
        }
        prefix = urgency.get(level, "Review and assess")
        return f"{prefix}: Implement corrective action for '{check_text}' per ICH E6(R3)."

    @staticmethod
    def _generate_summary(report: ComplianceReport) -> str:
        """Generate an executive summary for the report."""
        critical = sum(1 for f in report.findings if f.level == ComplianceLevel.CRITICAL_FINDING)
        major = sum(1 for f in report.findings if f.level == ComplianceLevel.MAJOR_FINDING)
        minor = sum(1 for f in report.findings if f.level == ComplianceLevel.MINOR_FINDING)

        return (
            f"GCP compliance assessment for trial {report.trial_id} "
            f"({report.trial_phase.value}): Overall score {report.overall_score:.1%}. "
            f"Findings: {critical} critical, {major} major, {minor} minor. "
            f"Assessment per ICH E6(R3) published September 2025."
        )

    @staticmethod
    def _generate_recommendations(report: ComplianceReport) -> list[str]:
        """Generate prioritized recommendations."""
        recs: list[str] = []

        critical = [f for f in report.findings if f.level == ComplianceLevel.CRITICAL_FINDING]
        if critical:
            recs.append(f"IMMEDIATE: Address {len(critical)} critical finding(s) before any further enrollment.")

        major = [f for f in report.findings if f.level == ComplianceLevel.MAJOR_FINDING]
        if major:
            recs.append(f"URGENT: Resolve {len(major)} major finding(s) within 7 days.")

        minor = [f for f in report.findings if f.level == ComplianceLevel.MINOR_FINDING]
        if minor:
            recs.append(f"ROUTINE: Address {len(minor)} minor finding(s) within 30 days.")

        if not report.findings:
            recs.append("No findings identified. Maintain current compliance posture.")

        recs.append("Schedule follow-up assessment per ICH E6(R3) risk-proportionate approach.")

        return recs

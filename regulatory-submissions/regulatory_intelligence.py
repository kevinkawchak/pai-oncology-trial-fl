#!/usr/bin/env python3
"""Regulatory intelligence engine -- guidance tracking and impact assessment.

RESEARCH USE ONLY -- Not intended for clinical deployment without validation.

This module implements a regulatory landscape monitoring and guidance tracking
engine for AI/ML-enabled medical devices across global regulatory authorities.
It tracks FDA, EMA, PMDA, TGA, Health Canada, MHRA, and NMPA guidance
documents, assesses their impact on ongoing submissions, generates compliance
timelines, monitors enforcement actions, and supports device classification
lookups.

All guidance data is simulated in-process.  No network calls or external API
dependencies.

Features:
    * Multi-jurisdiction guidance tracking (FDA, EMA, PMDA, etc.)
    * Impact assessment with severity scoring
    * Compliance timeline generation with milestone mapping
    * Enforcement action monitoring
    * Device classification lookups
    * Guidance change alerting

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
from datetime import datetime, timedelta, timezone
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

MAX_GUIDANCE_DOCUMENTS: int = 2000
MAX_UPDATES: int = 5000
MAX_ASSESSMENTS: int = 1000
MAX_TIMELINE_EVENTS: int = 500


# ===================================================================
# Enums
# ===================================================================
class Jurisdiction(str, Enum):
    """Regulatory authority jurisdictions."""

    FDA = "fda"
    EMA = "ema"
    PMDA = "pmda"
    TGA = "tga"
    HEALTH_CANADA = "health_canada"
    MHRA = "mhra"
    NMPA = "nmpa"


class GuidanceStatus(str, Enum):
    """Status of a guidance document."""

    DRAFT = "draft"
    FINAL = "final"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"


class ImpactLevel(str, Enum):
    """Impact level of a regulatory update on a submission."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class GuidanceCategory(str, Enum):
    """Categories of regulatory guidance documents."""

    AI_ML = "ai_ml"
    CYBERSECURITY = "cybersecurity"
    CLINICAL_EVIDENCE = "clinical_evidence"
    SOFTWARE = "software"
    RISK_MANAGEMENT = "risk_management"
    DATA_PRIVACY = "data_privacy"
    INTEROPERABILITY = "interoperability"
    POST_MARKET = "post_market"
    LABELING = "labeling"
    GENERAL_DEVICE = "general_device"


class EnforcementType(str, Enum):
    """Types of regulatory enforcement actions."""

    WARNING_LETTER = "warning_letter"
    RECALL = "recall"
    IMPORT_ALERT = "import_alert"
    CONSENT_DECREE = "consent_decree"
    CIVIL_PENALTY = "civil_penalty"
    UNTITLED_LETTER = "untitled_letter"


class DeviceClassification(str, Enum):
    """FDA device classification levels."""

    CLASS_I = "class_i"
    CLASS_II = "class_ii"
    CLASS_III = "class_iii"
    UNCLASSIFIED = "unclassified"


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class GuidanceDocument:
    """A regulatory guidance document."""

    guidance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    jurisdiction: Jurisdiction = Jurisdiction.FDA
    status: GuidanceStatus = GuidanceStatus.FINAL
    category: GuidanceCategory = GuidanceCategory.GENERAL_DEVICE
    publication_date: str = ""
    effective_date: str = ""
    comment_deadline: str = ""
    document_number: str = ""
    url: str = ""
    summary: str = ""
    key_requirements: List[str] = field(default_factory=list)
    applicable_device_types: List[str] = field(default_factory=list)
    supersedes: str = ""
    tags: List[str] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if the guidance is currently active."""
        return self.status in (GuidanceStatus.DRAFT, GuidanceStatus.FINAL)

    def days_since_publication(self, reference_date: Optional[str] = None) -> Optional[int]:
        """Days since publication."""
        if not self.publication_date:
            return None
        try:
            pub = datetime.fromisoformat(self.publication_date)
            ref = datetime.fromisoformat(reference_date) if reference_date else datetime.now(timezone.utc)
            return (ref - pub).days
        except (ValueError, TypeError):
            return None


@dataclass
class RegulatoryUpdate:
    """A regulatory update or notification."""

    update_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    jurisdiction: Jurisdiction = Jurisdiction.FDA
    update_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    impact_level: ImpactLevel = ImpactLevel.INFORMATIONAL
    category: GuidanceCategory = GuidanceCategory.GENERAL_DEVICE
    description: str = ""
    affected_submissions: List[str] = field(default_factory=list)
    action_required: str = ""
    deadline: str = ""
    source_guidance_id: str = ""


@dataclass
class ImpactAssessment:
    """Assessment of a regulatory update's impact on a submission."""

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    update_id: str = ""
    submission_id: str = ""
    assessed_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    impact_level: ImpactLevel = ImpactLevel.LOW
    impact_description: str = ""
    affected_sections: List[str] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)
    estimated_effort_days: int = 0
    risk_if_not_addressed: str = ""
    assessed_by: str = ""
    mitigation_plan: str = ""

    def impact_score(self) -> float:
        """Compute a numeric impact score (0-100)."""
        level_scores = {
            ImpactLevel.HIGH: 80.0,
            ImpactLevel.MEDIUM: 50.0,
            ImpactLevel.LOW: 20.0,
            ImpactLevel.INFORMATIONAL: 5.0,
        }
        base = level_scores.get(self.impact_level, 0.0)
        section_factor = min(len(self.affected_sections) * 5, 20)
        return min(base + section_factor, 100.0)


@dataclass
class ComplianceTimeline:
    """Timeline for compliance with a regulatory requirement."""

    timeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = ""
    guidance_id: str = ""
    title: str = ""
    milestones: List[Dict[str, str]] = field(default_factory=list)
    total_duration_days: int = 0
    start_date: str = ""
    end_date: str = ""
    is_on_track: bool = True
    progress_percentage: float = 0.0


@dataclass
class EnforcementAction:
    """A regulatory enforcement action."""

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    jurisdiction: Jurisdiction = Jurisdiction.FDA
    action_type: EnforcementType = EnforcementType.WARNING_LETTER
    date_issued: str = ""
    company_name: str = ""
    device_type: str = ""
    description: str = ""
    relevance_score: float = 0.0
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class ClassificationEntry:
    """A device classification database entry."""

    product_code: str = ""
    device_name: str = ""
    classification: DeviceClassification = DeviceClassification.CLASS_II
    regulation_number: str = ""
    submission_type: str = "510(k)"
    panel: str = ""
    definition: str = ""


# ===================================================================
# Built-in guidance database (simulated)
# ===================================================================
def _build_ai_ml_guidance_database() -> List[GuidanceDocument]:
    """Build a simulated database of AI/ML-related guidance documents."""
    return [
        GuidanceDocument(
            title="Artificial Intelligence and Machine Learning (AI/ML)-Enabled Device Functions",
            jurisdiction=Jurisdiction.FDA,
            status=GuidanceStatus.FINAL,
            category=GuidanceCategory.AI_ML,
            publication_date="2025-01-15",
            document_number="FDA-2019-N-5592",
            summary="Framework for AI/ML-based SaMD regulatory oversight.",
            key_requirements=[
                "Predetermined Change Control Plan (PCCP) required",
                "Good Machine Learning Practice (GMLP) principles",
                "Transparency and real-world performance monitoring",
            ],
            applicable_device_types=["SaMD", "AI/ML diagnostic", "CADx", "CADe"],
            tags=["ai", "ml", "samd", "pccp"],
        ),
        GuidanceDocument(
            title="Clinical Decision Support Software",
            jurisdiction=Jurisdiction.FDA,
            status=GuidanceStatus.FINAL,
            category=GuidanceCategory.SOFTWARE,
            publication_date="2022-09-28",
            document_number="FDA-2017-D-6569",
            summary="Criteria for CDS functions exempt from device regulation.",
            key_requirements=["Four-criteria test for CDS exemption", "Display requirements", "Transparency criteria"],
            applicable_device_types=["CDS", "SaMD"],
            tags=["cds", "software", "exemption"],
        ),
        GuidanceDocument(
            title="Cybersecurity in Medical Devices",
            jurisdiction=Jurisdiction.FDA,
            status=GuidanceStatus.FINAL,
            category=GuidanceCategory.CYBERSECURITY,
            publication_date="2023-09-27",
            document_number="FDA-2022-D-2921",
            summary="Premarket cybersecurity guidance for medical devices.",
            key_requirements=[
                "Secure Product Development Framework (SPDF)",
                "Software Bill of Materials (SBOM)",
                "Vulnerability management plan",
            ],
            applicable_device_types=["connected device", "SaMD", "network-capable"],
            tags=["cybersecurity", "sbom", "spdf"],
        ),
        GuidanceDocument(
            title="Predetermined Change Control Plans for ML-Enabled Device Software Functions",
            jurisdiction=Jurisdiction.FDA,
            status=GuidanceStatus.FINAL,
            category=GuidanceCategory.AI_ML,
            publication_date="2024-03-15",
            document_number="FDA-2023-D-0001",
            summary="Detailed guidance on PCCP content and structure.",
            key_requirements=[
                "SaMD Pre-Specifications (SPS)",
                "Algorithm Change Protocol (ACP)",
                "Performance monitoring commitments",
            ],
            applicable_device_types=["SaMD", "AI/ML"],
            tags=["pccp", "sps", "acp", "ai", "ml"],
        ),
        GuidanceDocument(
            title="Software as a Medical Device (SaMD): Clinical Evaluation",
            jurisdiction=Jurisdiction.FDA,
            status=GuidanceStatus.FINAL,
            category=GuidanceCategory.CLINICAL_EVIDENCE,
            publication_date="2017-12-08",
            document_number="IMDRF-SaMD-N41",
            summary="IMDRF framework for SaMD clinical evaluation.",
            key_requirements=["Clinical evaluation pathway", "Evidence categories", "Risk categorization"],
            applicable_device_types=["SaMD"],
            tags=["samd", "clinical", "imdrf"],
        ),
        GuidanceDocument(
            title="Artificial Intelligence in Medical Devices -- MDCG 2024-3",
            jurisdiction=Jurisdiction.EMA,
            status=GuidanceStatus.FINAL,
            category=GuidanceCategory.AI_ML,
            publication_date="2024-07-01",
            document_number="MDCG-2024-3",
            summary="EU MDR guidance on AI in medical devices.",
            key_requirements=[
                "Risk classification per MDR Annex VIII",
                "Clinical evaluation per MDR Article 61",
                "PMCF",
            ],
            applicable_device_types=["SaMD", "AI/ML", "medical device software"],
            tags=["mdr", "ai", "eu", "mdcg"],
        ),
        GuidanceDocument(
            title="AI/ML Medical Device Guidance",
            jurisdiction=Jurisdiction.PMDA,
            status=GuidanceStatus.FINAL,
            category=GuidanceCategory.AI_ML,
            publication_date="2024-01-15",
            document_number="PMDA-AI-2024-001",
            summary="PMDA guidance on regulatory review of AI/ML medical devices in Japan.",
            key_requirements=["STED documentation", "Clinical performance evaluation", "Change management protocol"],
            applicable_device_types=["SaMD", "AI/ML"],
            tags=["pmda", "japan", "ai", "ml"],
        ),
    ]


def _build_classification_database() -> List[ClassificationEntry]:
    """Build a simulated device classification database."""
    return [
        ClassificationEntry(
            "QAS", "Radiological CAD", DeviceClassification.CLASS_II, "892.2060", "510(k)", "Radiology"
        ),
        ClassificationEntry(
            "QDQ", "CADe Mammography", DeviceClassification.CLASS_II, "892.2060", "510(k)", "Radiology"
        ),
        ClassificationEntry(
            "QFP", "AI/ML Radiology Assist", DeviceClassification.CLASS_II, "892.2060", "510(k)", "Radiology"
        ),
        ClassificationEntry(
            "QMT", "Clinical Decision Support", DeviceClassification.CLASS_II, "892.2080", "De Novo", "Radiology"
        ),
        ClassificationEntry("OQO", "Oncology DSS", DeviceClassification.CLASS_II, "892.2090", "510(k)", "Radiology"),
        ClassificationEntry("QIH", "Pathology AI", DeviceClassification.CLASS_II, "864.3600", "De Novo", "Pathology"),
    ]


# ===================================================================
# Main class
# ===================================================================
class RegulatoryIntelligenceEngine:
    """Regulatory landscape monitoring and guidance intelligence engine.

    Tracks guidance documents across FDA, EMA, PMDA, and other authorities,
    assesses impact on ongoing submissions, generates compliance timelines,
    and monitors enforcement actions.

    Parameters
    ----------
    submission_id : str
        Submission identifier for context.
    jurisdictions : Sequence[Jurisdiction], optional
        Jurisdictions to monitor.  Defaults to all.
    """

    def __init__(
        self,
        submission_id: str = "",
        jurisdictions: Optional[Sequence[Jurisdiction]] = None,
    ) -> None:
        self._submission_id = submission_id
        self._jurisdictions = list(jurisdictions) if jurisdictions else list(Jurisdiction)
        self._guidance_db: Dict[str, GuidanceDocument] = {}
        self._updates: Dict[str, RegulatoryUpdate] = {}
        self._assessments: Dict[str, ImpactAssessment] = {}
        self._enforcement_actions: Dict[str, EnforcementAction] = {}
        self._classification_db: List[ClassificationEntry] = _build_classification_database()
        self._load_guidance_database()
        logger.info(
            "RegulatoryIntelligenceEngine initialized: %d guidance docs, %d jurisdictions",
            len(self._guidance_db),
            len(self._jurisdictions),
        )

    def _load_guidance_database(self) -> None:
        """Load built-in guidance database filtered by jurisdictions."""
        for doc in _build_ai_ml_guidance_database():
            if doc.jurisdiction in self._jurisdictions:
                self._guidance_db[doc.guidance_id] = doc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def submission_id(self) -> str:
        return self._submission_id

    @property
    def guidance_count(self) -> int:
        return len(self._guidance_db)

    @property
    def update_count(self) -> int:
        return len(self._updates)

    @property
    def jurisdictions(self) -> List[Jurisdiction]:
        return list(self._jurisdictions)

    # ------------------------------------------------------------------
    # Guidance access
    # ------------------------------------------------------------------
    def get_guidance_by_jurisdiction(self, jurisdiction: Jurisdiction) -> List[GuidanceDocument]:
        """Return all guidance documents for a jurisdiction."""
        return [g for g in self._guidance_db.values() if g.jurisdiction == jurisdiction]

    def get_guidance_by_category(self, category: GuidanceCategory) -> List[GuidanceDocument]:
        """Return all guidance documents in a category."""
        return [g for g in self._guidance_db.values() if g.category == category]

    def get_active_guidance(self) -> List[GuidanceDocument]:
        """Return all currently active guidance documents."""
        return [g for g in self._guidance_db.values() if g.is_active()]

    def search_guidance(self, query: str) -> List[GuidanceDocument]:
        """Search guidance documents by keyword in title, summary, or tags."""
        query_lower = query.lower()
        results = []
        for g in self._guidance_db.values():
            if (
                query_lower in g.title.lower()
                or query_lower in g.summary.lower()
                or any(query_lower in tag.lower() for tag in g.tags)
            ):
                results.append(g)
        return results

    def add_guidance(self, guidance: GuidanceDocument) -> None:
        """Add a guidance document to the tracking database."""
        if len(self._guidance_db) >= MAX_GUIDANCE_DOCUMENTS:
            logger.warning("Maximum guidance document capacity reached")
            return
        self._guidance_db[guidance.guidance_id] = guidance
        logger.info("Guidance added: %s (%s)", guidance.title, guidance.jurisdiction.value)

    # ------------------------------------------------------------------
    # Impact assessment
    # ------------------------------------------------------------------
    def assess_impact(
        self,
        guidance_id: str,
        submission_id: Optional[str] = None,
        affected_sections: Optional[List[str]] = None,
        assessed_by: str = "system",
    ) -> Optional[ImpactAssessment]:
        """Assess the impact of a guidance document on a submission."""
        guidance = self._guidance_db.get(guidance_id)
        if guidance is None:
            logger.warning("Guidance not found: %s", guidance_id)
            return None

        sub_id = submission_id or self._submission_id
        sections = affected_sections or []

        # Determine impact level based on guidance characteristics
        if guidance.status == GuidanceStatus.FINAL and guidance.category == GuidanceCategory.AI_ML:
            impact = ImpactLevel.HIGH
        elif guidance.status == GuidanceStatus.FINAL:
            impact = ImpactLevel.MEDIUM
        elif guidance.status == GuidanceStatus.DRAFT:
            impact = ImpactLevel.LOW
        else:
            impact = ImpactLevel.INFORMATIONAL

        required_changes = []
        for req in guidance.key_requirements:
            required_changes.append(f"Review and address: {req}")

        effort = len(required_changes) * 5
        if impact == ImpactLevel.HIGH:
            effort = int(effort * 1.5)

        assessment = ImpactAssessment(
            update_id=guidance_id,
            submission_id=sub_id,
            impact_level=impact,
            impact_description=f"Impact of '{guidance.title}' on submission {sub_id}",
            affected_sections=sections
            or [
                s
                for s in ["Module 2", "Module 5"]
                if guidance.category in (GuidanceCategory.AI_ML, GuidanceCategory.CLINICAL_EVIDENCE)
            ],
            required_changes=required_changes,
            estimated_effort_days=min(effort, 180),
            risk_if_not_addressed=f"Non-compliance with {guidance.jurisdiction.value} {guidance.document_number}",
            assessed_by=assessed_by,
        )
        self._assessments[assessment.assessment_id] = assessment
        logger.info("Impact assessed: %s -> %s (level=%s)", guidance.title[:40], sub_id, impact.value)
        return assessment

    def get_high_impact_items(self) -> List[ImpactAssessment]:
        """Return all high-impact assessments."""
        return [a for a in self._assessments.values() if a.impact_level == ImpactLevel.HIGH]

    # ------------------------------------------------------------------
    # Compliance timelines
    # ------------------------------------------------------------------
    def generate_compliance_timeline(
        self,
        guidance_id: str,
        start_date: Optional[str] = None,
        submission_id: Optional[str] = None,
    ) -> Optional[ComplianceTimeline]:
        """Generate a compliance timeline for a guidance document."""
        guidance = self._guidance_db.get(guidance_id)
        if guidance is None:
            logger.warning("Guidance not found: %s", guidance_id)
            return None

        start = start_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sub_id = submission_id or self._submission_id

        milestones = [
            {"name": "Gap analysis", "offset_days": "0", "duration_days": "14"},
            {"name": "Impact assessment", "offset_days": "14", "duration_days": "7"},
            {"name": "Remediation planning", "offset_days": "21", "duration_days": "14"},
            {"name": "Documentation updates", "offset_days": "35", "duration_days": "30"},
            {"name": "Internal review", "offset_days": "65", "duration_days": "14"},
            {"name": "Final compliance verification", "offset_days": "79", "duration_days": "7"},
        ]

        total_days = 86
        try:
            start_dt = datetime.fromisoformat(start)
            end_date = (start_dt + timedelta(days=total_days)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            end_date = ""

        timeline = ComplianceTimeline(
            submission_id=sub_id,
            guidance_id=guidance_id,
            title=f"Compliance timeline: {guidance.title[:60]}",
            milestones=milestones,
            total_duration_days=total_days,
            start_date=start,
            end_date=end_date,
        )
        logger.info("Timeline generated: %d days for %s", total_days, guidance.title[:40])
        return timeline

    # ------------------------------------------------------------------
    # Enforcement actions
    # ------------------------------------------------------------------
    def add_enforcement_action(self, action: EnforcementAction) -> None:
        """Track an enforcement action."""
        self._enforcement_actions[action.action_id] = action
        logger.info("Enforcement action tracked: %s", action.title)

    def get_relevant_enforcement_actions(self, device_type: str = "") -> List[EnforcementAction]:
        """Return enforcement actions relevant to a device type."""
        if not device_type:
            return list(self._enforcement_actions.values())
        return [
            a
            for a in self._enforcement_actions.values()
            if device_type.lower() in a.device_type.lower() or a.relevance_score > 0.5
        ]

    # ------------------------------------------------------------------
    # Device classification
    # ------------------------------------------------------------------
    def lookup_classification(self, product_code: str = "", device_name: str = "") -> List[ClassificationEntry]:
        """Look up device classification by product code or name."""
        results = []
        for entry in self._classification_db:
            if product_code and entry.product_code.upper() == product_code.upper():
                results.append(entry)
            elif device_name and device_name.lower() in entry.device_name.lower():
                results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_intelligence_report(self) -> str:
        """Generate a Markdown regulatory intelligence report."""
        lines: List[str] = [
            "# Regulatory Intelligence Report",
            "",
            f"**Submission:** {self._submission_id}",
            f"**Jurisdictions:** {', '.join(j.value.upper() for j in self._jurisdictions)}",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "",
            "## Guidance Document Landscape",
            "",
        ]

        for jurisdiction in self._jurisdictions:
            docs = self.get_guidance_by_jurisdiction(jurisdiction)
            if docs:
                lines.append(f"### {jurisdiction.value.upper()} ({len(docs)} documents)")
                lines.append("")
                for doc in docs:
                    status_icon = "ACTIVE" if doc.is_active() else doc.status.value.upper()
                    lines.append(f"- [{status_icon}] **{doc.title}** ({doc.publication_date})")
                    if doc.key_requirements:
                        for req in doc.key_requirements[:3]:
                            lines.append(f"  - {req}")
                lines.append("")

        # Impact summary
        if self._assessments:
            lines.extend(["## Impact Assessments", ""])
            lines.append("| Guidance | Impact | Effort (days) | Sections Affected |")
            lines.append("|----------|--------|--------------|-------------------|")
            for assessment in self._assessments.values():
                sections = ", ".join(assessment.affected_sections[:3]) or "N/A"
                lines.append(
                    f"| {assessment.impact_description[:40]} | "
                    f"{assessment.impact_level.value} | "
                    f"{assessment.estimated_effort_days} | {sections} |"
                )
            lines.append("")

        # Classification reference
        lines.extend(["## Device Classification Reference", ""])
        lines.append("| Product Code | Device Name | Class | Submission |")
        lines.append("|-------------|-------------|-------|------------|")
        for entry in self._classification_db[:6]:
            lines.append(
                f"| {entry.product_code} | {entry.device_name} | "
                f"{entry.classification.value} | {entry.submission_type} |"
            )

        lines.extend(["", f"*Report generated: {datetime.now(timezone.utc).isoformat()[:19]}*"])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state to a dictionary."""
        return {
            "submission_id": self._submission_id,
            "jurisdictions": [j.value for j in self._jurisdictions],
            "guidance_count": len(self._guidance_db),
            "update_count": len(self._updates),
            "assessment_count": len(self._assessments),
            "enforcement_count": len(self._enforcement_actions),
        }


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = RegulatoryIntelligenceEngine("SUB-2026-DEMO", [Jurisdiction.FDA, Jurisdiction.EMA])
    report = engine.generate_intelligence_report()
    print(report)

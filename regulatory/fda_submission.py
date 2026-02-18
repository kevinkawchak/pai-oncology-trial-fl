"""FDA submission tracking for AI/ML-enabled medical devices.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Manages the regulatory submission lifecycle for AI-powered oncology
devices, covering 510(k), De Novo, PMA, and Pre-Submission pathways.
Tracks required documentation based on the January 2025 FDA draft
guidance on AI/ML-enabled devices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class SubmissionPathway(str, Enum):
    """FDA submission pathways for medical devices."""

    PRESUBMISSION = "pre_submission"
    DENOVO = "de_novo"
    K510 = "510k"
    PMA = "pma"
    BREAKTHROUGH = "breakthrough_device"


class DocumentCategory(str, Enum):
    """Categories of required FDA submission documents."""

    DEVICE_DESCRIPTION = "device_description"
    AI_ML_DOCUMENTATION = "ai_ml_documentation"
    CLINICAL_EVIDENCE = "clinical_evidence"
    HUMAN_FACTORS = "human_factors"
    CYBERSECURITY = "cybersecurity"
    QUALITY_SYSTEM = "quality_system"
    POST_MARKET = "post_market_surveillance"
    SOFTWARE_LIFECYCLE = "software_lifecycle"


class DocumentStatus(str, Enum):
    """Status of a submission document."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    REVIEW = "in_review"
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"


@dataclass
class SubmissionDocument:
    """A single document in the submission package.

    Attributes:
        doc_id: Unique document identifier.
        category: Document category.
        title: Human-readable title.
        status: Current preparation status.
        notes: Reviewer notes or comments.
    """

    doc_id: str
    category: DocumentCategory
    title: str
    status: DocumentStatus = DocumentStatus.NOT_STARTED
    notes: str = ""


@dataclass
class SubmissionPackage:
    """Complete FDA submission package for an AI/ML device.

    Attributes:
        submission_id: Unique submission identifier.
        pathway: Selected regulatory pathway.
        device_name: Name of the AI/ML device.
        indication: Intended use / indication for use.
        documents: Required documents.
        created_at: ISO timestamp.
        submitted_at: When the package was submitted to FDA.
        milestones: Key dates and events.
    """

    submission_id: str
    pathway: SubmissionPathway
    device_name: str
    indication: str = ""
    documents: list[SubmissionDocument] = field(default_factory=list)
    created_at: str = ""
    submitted_at: str | None = None
    milestones: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# Standard document requirements per pathway
_STANDARD_DOCS: dict[SubmissionPathway, list[tuple[DocumentCategory, str]]] = {
    SubmissionPathway.K510: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Intended Use"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Algorithm Description"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Training Data Description and Provenance"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Model Performance Evaluation"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Clinical Evidence Summary"),
        (DocumentCategory.HUMAN_FACTORS, "Human Factors Validation"),
        (DocumentCategory.CYBERSECURITY, "Cybersecurity Documentation"),
        (DocumentCategory.SOFTWARE_LIFECYCLE, "IEC 62304 Software Lifecycle Plan"),
        (DocumentCategory.QUALITY_SYSTEM, "Quality Management System Summary"),
    ],
    SubmissionPathway.DENOVO: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Classification"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Algorithm Description"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Training Data Description and Provenance"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Model Performance — Standalone and Clinical"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Predetermined Change Control Plan (PCCP)"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Clinical Evidence Package"),
        (DocumentCategory.HUMAN_FACTORS, "Human Factors and Usability Study"),
        (DocumentCategory.CYBERSECURITY, "Cybersecurity Risk Assessment"),
        (DocumentCategory.SOFTWARE_LIFECYCLE, "IEC 62304 Software Documentation"),
        (DocumentCategory.POST_MARKET, "Post-Market Surveillance Plan"),
        (DocumentCategory.QUALITY_SYSTEM, "Quality Management System"),
    ],
    SubmissionPathway.PMA: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Principles of Operation"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Algorithm Detailed Documentation"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Training Dataset and Bias Analysis"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Predetermined Change Control Plan (PCCP)"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Pivotal Clinical Trial Results"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Benefit-Risk Analysis"),
        (DocumentCategory.HUMAN_FACTORS, "Human Factors Engineering Report"),
        (DocumentCategory.CYBERSECURITY, "Cybersecurity Plan and Threat Model"),
        (DocumentCategory.SOFTWARE_LIFECYCLE, "Complete Software Lifecycle Documentation"),
        (DocumentCategory.POST_MARKET, "Post-Market Surveillance and REMS Plan"),
        (DocumentCategory.QUALITY_SYSTEM, "Complete Quality System Documentation"),
    ],
    SubmissionPathway.PRESUBMISSION: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Concept Description"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Proposed AI/ML Approach"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Proposed Clinical Evaluation Plan"),
    ],
    SubmissionPathway.BREAKTHROUGH: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Breakthrough Justification"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Innovation Description"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Preliminary Clinical Evidence"),
    ],
}


class FDASubmissionTracker:
    """Tracks FDA submission packages and their documentation status.

    Creates submission packages with the standard required documents
    for each pathway and tracks their preparation progress.
    """

    def __init__(self) -> None:
        self._packages: dict[str, SubmissionPackage] = {}

    def create_submission(
        self,
        submission_id: str,
        pathway: SubmissionPathway | str,
        device_name: str,
        indication: str = "",
    ) -> SubmissionPackage:
        """Create a new submission package with standard documents.

        Args:
            submission_id: Unique identifier.
            pathway: Regulatory pathway.
            device_name: Name of the device.
            indication: Intended use statement.

        Returns:
            The created SubmissionPackage.
        """
        if isinstance(pathway, str):
            pathway = SubmissionPathway(pathway)

        docs = []
        for i, (cat, title) in enumerate(_STANDARD_DOCS.get(pathway, [])):
            docs.append(
                SubmissionDocument(
                    doc_id=f"{submission_id}_DOC_{i + 1:02d}",
                    category=cat,
                    title=title,
                )
            )

        pkg = SubmissionPackage(
            submission_id=submission_id,
            pathway=pathway,
            device_name=device_name,
            indication=indication,
            documents=docs,
        )
        self._packages[submission_id] = pkg
        logger.info(
            "Created %s submission '%s' with %d documents",
            pathway.value,
            submission_id,
            len(docs),
        )
        return pkg

    def update_document_status(
        self,
        submission_id: str,
        doc_id: str,
        status: DocumentStatus | str,
        notes: str = "",
    ) -> bool:
        """Update the status of a document in a submission package."""
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return False
        if isinstance(status, str):
            status = DocumentStatus(status)

        for doc in pkg.documents:
            if doc.doc_id == doc_id:
                doc.status = status
                if notes:
                    doc.notes = notes
                return True
        return False

    def add_milestone(self, submission_id: str, milestone: str, date: str = "") -> bool:
        """Record a milestone for a submission."""
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return False
        pkg.milestones.append(
            {
                "milestone": milestone,
                "date": date or datetime.now(timezone.utc).isoformat(),
            }
        )
        return True

    def get_submission(self, submission_id: str) -> SubmissionPackage | None:
        """Retrieve a submission package."""
        return self._packages.get(submission_id)

    def get_readiness_report(self, submission_id: str) -> dict:
        """Generate a readiness report for a submission.

        Returns a summary of document statuses and an overall
        readiness assessment.
        """
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return {"error": "Submission not found"}

        status_counts: dict[str, int] = {}
        for doc in pkg.documents:
            s = doc.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        total = len(pkg.documents)
        approved = status_counts.get("approved", 0)
        ready = approved == total and total > 0

        return {
            "submission_id": submission_id,
            "pathway": pkg.pathway.value,
            "device_name": pkg.device_name,
            "total_documents": total,
            "by_status": status_counts,
            "ready_for_submission": ready,
            "milestones": pkg.milestones,
        }

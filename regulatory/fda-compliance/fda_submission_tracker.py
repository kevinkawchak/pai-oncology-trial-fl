"""FDA submission lifecycle tracker for AI/ML-enabled medical devices.

Manages the regulatory submission workflow for AI-powered oncology
devices across all FDA pathways: 510(k), De Novo, PMA, Pre-Submission,
and Breakthrough Device Designation. Tracks required documentation,
milestones, and review cycles per current FDA guidance.

References:
    - FDA Draft Guidance: Marketing Submission Recommendations for a
      Predetermined Change Control Plan for AI/ML-Enabled Device
      Software Functions (January 2025)
    - FDA PCCP Guidance: Predetermined Change Control Plans for
      Machine Learning-Enabled Device Software Functions (August 2025)
    - FDA QMSR: Quality Management System Regulation, 21 CFR Part 820
      (effective February 2026)
    - 21 CFR Part 11: Electronic Records; Electronic Signatures
    - IEC 62304: Medical device software — Software life cycle processes
    - ISO 14971:2019: Medical devices — Application of risk management

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SubmissionType(str, Enum):
    """FDA submission pathways for medical devices.

    Per FDA guidance, AI/ML-enabled devices may require
    different levels of evidence depending on the pathway.
    """

    K510 = "510k"
    DE_NOVO = "de_novo"
    PMA = "pma"
    BREAKTHROUGH = "breakthrough_device"
    PRESUBMISSION = "pre_submission"


class SubmissionStatus(str, Enum):
    """Lifecycle status of an FDA submission."""

    PLANNING = "planning"
    DRAFTING = "drafting"
    INTERNAL_REVIEW = "internal_review"
    SUBMITTED = "submitted"
    FDA_REVIEW = "fda_review"
    ADDITIONAL_INFO_REQUESTED = "additional_info_requested"
    ACCEPTED = "accepted"
    CLEARED = "cleared"  # 510(k)
    GRANTED = "granted"  # De Novo
    APPROVED = "approved"  # PMA
    DENIED = "denied"
    WITHDRAWN = "withdrawn"


class DeviceClass(str, Enum):
    """FDA device classification.

    AI/ML-enabled devices are typically Class II (510(k))
    or Class III (PMA), with some De Novo Class II pathways.
    """

    CLASS_I = "class_i"
    CLASS_II = "class_ii"
    CLASS_III = "class_iii"


class DocumentCategory(str, Enum):
    """Categories of submission documents."""

    DEVICE_DESCRIPTION = "device_description"
    INTENDED_USE = "intended_use"
    AI_ML_DOCUMENTATION = "ai_ml_documentation"
    TRAINING_DATA = "training_data"
    PERFORMANCE_TESTING = "performance_testing"
    CLINICAL_EVIDENCE = "clinical_evidence"
    HUMAN_FACTORS = "human_factors"
    CYBERSECURITY = "cybersecurity"
    SOFTWARE_LIFECYCLE = "software_lifecycle"
    RISK_MANAGEMENT = "risk_management"
    QUALITY_SYSTEM = "quality_system"
    LABELING = "labeling"
    POST_MARKET = "post_market_surveillance"
    PCCP = "predetermined_change_control_plan"
    BIOCOMPATIBILITY = "biocompatibility"


class DocumentStatus(str, Enum):
    """Preparation status of a submission document."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DRAFT_COMPLETE = "draft_complete"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    FINALIZED = "finalized"


class ReviewCycleType(str, Enum):
    """Types of FDA review interactions."""

    INITIAL_REVIEW = "initial_review"
    ADDITIONAL_INFO = "additional_info_request"
    DEFICIENCY_LETTER = "deficiency_letter"
    INTERACTIVE_REVIEW = "interactive_review"
    PANEL_REVIEW = "panel_review"
    FINAL_DECISION = "final_decision"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AIMLComponent:
    """Description of an AI/ML component within the device.

    Attributes:
        component_id: Unique component identifier.
        name: Component name.
        model_type: Type of ML model (default: "unspecified").
        input_types: Types of input data.
        output_types: Types of output predictions.
        training_data_size: Size of training dataset.
        intended_population: Target patient population.
        locked_or_adaptive: Whether the algorithm is locked or adaptive.
        pccp_applicable: Whether a PCCP is applicable.
    """

    component_id: str
    name: str
    model_type: str = "unspecified"
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    training_data_size: int = 0
    intended_population: str = ""
    locked_or_adaptive: str = "locked"
    pccp_applicable: bool = False


@dataclass
class SubmissionDocument:
    """A document in the submission package.

    Attributes:
        doc_id: Unique document identifier.
        category: Document category.
        title: Document title.
        status: Preparation status.
        version: Document version.
        author: Primary author.
        reviewer: Assigned reviewer.
        notes: Reviewer notes.
        last_updated: Last modification timestamp.
    """

    doc_id: str
    category: DocumentCategory
    title: str
    status: DocumentStatus = DocumentStatus.NOT_STARTED
    version: str = "1.0"
    author: str = ""
    reviewer: str = ""
    notes: str = ""
    last_updated: str = ""

    def __post_init__(self) -> None:
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()


@dataclass
class ReviewCycle:
    """A review interaction with FDA.

    Attributes:
        cycle_id: Unique cycle identifier.
        cycle_type: Type of review interaction.
        initiated_at: When the cycle started.
        response_due: Deadline for response.
        completed_at: When the cycle was completed.
        findings: FDA findings or questions.
        response_summary: Summary of response provided.
    """

    cycle_id: str
    cycle_type: ReviewCycleType
    initiated_at: str = ""
    response_due: str = ""
    completed_at: str = ""
    findings: list[str] = field(default_factory=list)
    response_summary: str = ""

    def __post_init__(self) -> None:
        if not self.initiated_at:
            self.initiated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SubmissionPackage:
    """Complete FDA submission package.

    Attributes:
        submission_id: Unique identifier.
        submission_type: Regulatory pathway.
        status: Current lifecycle status.
        device_name: Name of the device.
        device_class: FDA device classification.
        indication: Intended use / indication for use.
        ai_ml_components: AI/ML components in the device.
        documents: Required documents.
        review_cycles: FDA review interactions.
        milestones: Key dates and events.
        created_at: Package creation timestamp.
        submitted_at: FDA submission timestamp.
        decision_date: FDA decision date.
        fda_reference_number: FDA-assigned reference number.
        predicate_device: Predicate device (for 510(k)).
        product_code: FDA product code.
        regulatory_contact: Primary regulatory contact.
    """

    submission_id: str
    submission_type: SubmissionType
    status: SubmissionStatus = SubmissionStatus.PLANNING
    device_name: str = ""
    device_class: DeviceClass = DeviceClass.CLASS_II
    indication: str = ""
    ai_ml_components: list[AIMLComponent] = field(default_factory=list)
    documents: list[SubmissionDocument] = field(default_factory=list)
    review_cycles: list[ReviewCycle] = field(default_factory=list)
    milestones: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    submitted_at: str = ""
    decision_date: str = ""
    fda_reference_number: str = ""
    predicate_device: str = ""
    product_code: str = ""
    regulatory_contact: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Standard document requirements per pathway
# ---------------------------------------------------------------------------

_STANDARD_DOCS: dict[SubmissionType, list[tuple[DocumentCategory, str]]] = {
    SubmissionType.K510: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Intended Use"),
        (DocumentCategory.INTENDED_USE, "Indications for Use Statement"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Algorithm Description"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Algorithm Change Protocol (ACP)"),
        (DocumentCategory.TRAINING_DATA, "Training Data Description and Provenance"),
        (DocumentCategory.TRAINING_DATA, "Data Representativeness Analysis"),
        (DocumentCategory.PERFORMANCE_TESTING, "Standalone Performance Testing"),
        (DocumentCategory.PERFORMANCE_TESTING, "Clinical Performance Testing"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Clinical Evidence Summary — Substantial Equivalence"),
        (DocumentCategory.HUMAN_FACTORS, "Human Factors Validation Report"),
        (DocumentCategory.CYBERSECURITY, "Cybersecurity Documentation per FDA Guidance"),
        (DocumentCategory.SOFTWARE_LIFECYCLE, "IEC 62304 Software Lifecycle Documentation"),
        (DocumentCategory.RISK_MANAGEMENT, "ISO 14971 Risk Management File"),
        (DocumentCategory.QUALITY_SYSTEM, "QMSR Compliance Summary (21 CFR Part 820)"),
        (DocumentCategory.LABELING, "Proposed Labeling"),
    ],
    SubmissionType.DE_NOVO: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Classification Rationale"),
        (DocumentCategory.INTENDED_USE, "Indications for Use Statement"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Algorithm Description — Novel Device"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Algorithm Change Protocol (ACP)"),
        (DocumentCategory.PCCP, "Predetermined Change Control Plan (PCCP) — Aug 2025 Guidance"),
        (DocumentCategory.TRAINING_DATA, "Training Data Description and Provenance"),
        (DocumentCategory.TRAINING_DATA, "Data Representativeness and Bias Analysis"),
        (DocumentCategory.PERFORMANCE_TESTING, "Standalone Performance Testing"),
        (DocumentCategory.PERFORMANCE_TESTING, "Clinical Performance Testing"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Clinical Evidence Package — De Novo"),
        (DocumentCategory.HUMAN_FACTORS, "Human Factors and Usability Engineering Report"),
        (DocumentCategory.CYBERSECURITY, "Cybersecurity Risk Assessment and Mitigation"),
        (DocumentCategory.SOFTWARE_LIFECYCLE, "IEC 62304 Software Documentation"),
        (DocumentCategory.RISK_MANAGEMENT, "ISO 14971 Risk Management File"),
        (DocumentCategory.POST_MARKET, "Post-Market Surveillance Plan"),
        (DocumentCategory.QUALITY_SYSTEM, "QMSR Compliance Summary"),
        (DocumentCategory.LABELING, "Proposed Labeling and IFU"),
    ],
    SubmissionType.PMA: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Principles of Operation"),
        (DocumentCategory.INTENDED_USE, "Indications for Use Statement"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Algorithm Detailed Documentation"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Algorithm Change Protocol (ACP)"),
        (DocumentCategory.PCCP, "Predetermined Change Control Plan (PCCP)"),
        (DocumentCategory.TRAINING_DATA, "Complete Training Dataset Documentation"),
        (DocumentCategory.TRAINING_DATA, "Bias Analysis and Mitigation Report"),
        (DocumentCategory.PERFORMANCE_TESTING, "Bench Testing Report"),
        (DocumentCategory.PERFORMANCE_TESTING, "Analytical Validation Report"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Pivotal Clinical Trial Results"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Benefit-Risk Analysis"),
        (DocumentCategory.HUMAN_FACTORS, "Human Factors Engineering Report"),
        (DocumentCategory.CYBERSECURITY, "Cybersecurity Plan and Threat Model"),
        (DocumentCategory.SOFTWARE_LIFECYCLE, "Complete IEC 62304 Software Lifecycle"),
        (DocumentCategory.RISK_MANAGEMENT, "ISO 14971 Risk Management Report"),
        (DocumentCategory.POST_MARKET, "Post-Market Surveillance and REMS Plan"),
        (DocumentCategory.QUALITY_SYSTEM, "Complete QMSR Documentation"),
        (DocumentCategory.LABELING, "Final Labeling Package"),
        (DocumentCategory.BIOCOMPATIBILITY, "Biocompatibility Assessment (if applicable)"),
    ],
    SubmissionType.PRESUBMISSION: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Concept Description"),
        (DocumentCategory.INTENDED_USE, "Proposed Indications for Use"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "Proposed AI/ML Approach Summary"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Proposed Clinical Evaluation Plan"),
        (DocumentCategory.RISK_MANAGEMENT, "Preliminary Risk Assessment"),
    ],
    SubmissionType.BREAKTHROUGH: [
        (DocumentCategory.DEVICE_DESCRIPTION, "Device Description and Breakthrough Justification"),
        (DocumentCategory.INTENDED_USE, "Proposed Indications for Use"),
        (DocumentCategory.AI_ML_DOCUMENTATION, "AI/ML Innovation Description"),
        (DocumentCategory.CLINICAL_EVIDENCE, "Preliminary Clinical Evidence"),
        (DocumentCategory.RISK_MANAGEMENT, "Preliminary Benefit-Risk Assessment"),
    ],
}

# Typical review timelines (calendar days) — informational only
_REVIEW_TIMELINES: dict[SubmissionType, int] = {
    SubmissionType.K510: 90,
    SubmissionType.DE_NOVO: 150,
    SubmissionType.PMA: 180,
    SubmissionType.PRESUBMISSION: 75,
    SubmissionType.BREAKTHROUGH: 60,
}


# ---------------------------------------------------------------------------
# FDA Submission Tracker
# ---------------------------------------------------------------------------


class FDASubmissionTracker:
    """Tracks FDA submission packages and their documentation lifecycle.

    Creates submission packages with standard required documents per
    pathway, tracks document preparation, manages review cycles with
    FDA, and generates readiness reports.

    References current FDA guidance:
    - AI/ML Device Guidance (January 2025 draft)
    - PCCP Guidance (August 2025)
    - QMSR (effective February 2026)
    """

    def __init__(self) -> None:
        self._packages: dict[str, SubmissionPackage] = {}
        self._audit_trail: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Package management
    # ------------------------------------------------------------------

    def create_submission(
        self,
        submission_id: str,
        submission_type: SubmissionType | str,
        device_name: str,
        device_class: DeviceClass | str = DeviceClass.CLASS_II,
        indication: str = "",
        predicate_device: str = "",
    ) -> SubmissionPackage:
        """Create a new submission package with standard documents.

        Args:
            submission_id: Unique identifier.
            submission_type: Regulatory pathway.
            device_name: Name of the device.
            device_class: FDA device classification.
            indication: Intended use statement.
            predicate_device: Predicate device (for 510(k)).

        Returns:
            The created SubmissionPackage.
        """
        if isinstance(submission_type, str):
            submission_type = SubmissionType(submission_type)
        if isinstance(device_class, str):
            device_class = DeviceClass(device_class)

        # Generate standard documents
        docs: list[SubmissionDocument] = []
        for i, (cat, title) in enumerate(_STANDARD_DOCS.get(submission_type, [])):
            docs.append(
                SubmissionDocument(
                    doc_id=f"{submission_id}_DOC_{i + 1:02d}",
                    category=cat,
                    title=title,
                )
            )

        pkg = SubmissionPackage(
            submission_id=submission_id,
            submission_type=submission_type,
            device_name=device_name,
            device_class=device_class,
            indication=indication,
            documents=docs,
            predicate_device=predicate_device,
        )
        self._packages[submission_id] = pkg

        self._log_audit(
            submission_id,
            "submission_created",
            f"type={submission_type.value}, device={device_name}, docs={len(docs)}",
        )
        logger.info(
            "Created %s submission '%s' with %d documents",
            submission_type.value,
            submission_id,
            len(docs),
        )
        return pkg

    def add_ai_ml_component(
        self,
        submission_id: str,
        component_id: str,
        name: str,
        model_type: str = "unspecified",
        input_types: list[str] | None = None,
        output_types: list[str] | None = None,
        locked_or_adaptive: str = "locked",
    ) -> bool:
        """Add an AI/ML component description to a submission.

        Note: model_type defaults to "unspecified" per best practice —
        the specific model type should be determined during validation.

        Args:
            submission_id: Submission to add the component to.
            component_id: Unique component identifier.
            name: Component name.
            model_type: Type of ML model (default: "unspecified").
            input_types: Types of input data.
            output_types: Types of output predictions.
            locked_or_adaptive: Whether the algorithm is locked or adaptive.

        Returns:
            True if the component was added successfully.
        """
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return False

        component = AIMLComponent(
            component_id=component_id,
            name=name,
            model_type=model_type,
            input_types=list(input_types or []),
            output_types=list(output_types or []),
            locked_or_adaptive=locked_or_adaptive,
            pccp_applicable=locked_or_adaptive == "adaptive",
        )
        pkg.ai_ml_components.append(component)
        self._log_audit(submission_id, "ai_ml_component_added", f"component={component_id}")
        return True

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def update_document_status(
        self,
        submission_id: str,
        doc_id: str,
        status: DocumentStatus | str,
        notes: str = "",
    ) -> bool:
        """Update the status of a submission document."""
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return False
        if isinstance(status, str):
            status = DocumentStatus(status)

        for doc in pkg.documents:
            if doc.doc_id == doc_id:
                old_status = doc.status
                doc.status = status
                doc.last_updated = datetime.now(timezone.utc).isoformat()
                if notes:
                    doc.notes = notes
                self._log_audit(
                    submission_id,
                    "document_status_updated",
                    f"doc={doc_id}, {old_status.value} -> {status.value}",
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Review cycles
    # ------------------------------------------------------------------

    def add_review_cycle(
        self,
        submission_id: str,
        cycle_type: ReviewCycleType | str,
        findings: list[str] | None = None,
        response_due_days: int = 180,
    ) -> str | None:
        """Add a review cycle (FDA interaction) to a submission.

        Args:
            submission_id: Target submission.
            cycle_type: Type of review interaction.
            findings: FDA findings or questions.
            response_due_days: Days until response is due.

        Returns:
            The cycle ID, or None if the submission was not found.
        """
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return None
        if isinstance(cycle_type, str):
            cycle_type = ReviewCycleType(cycle_type)

        cycle_id = f"{submission_id}_RC_{len(pkg.review_cycles) + 1:02d}"
        now = datetime.now(timezone.utc)
        due = (now + timedelta(days=response_due_days)).isoformat()

        cycle = ReviewCycle(
            cycle_id=cycle_id,
            cycle_type=cycle_type,
            response_due=due,
            findings=list(findings or []),
        )
        pkg.review_cycles.append(cycle)
        self._log_audit(submission_id, "review_cycle_added", f"type={cycle_type.value}")
        return cycle_id

    def update_submission_status(
        self,
        submission_id: str,
        status: SubmissionStatus | str,
    ) -> bool:
        """Update the overall submission status."""
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return False
        if isinstance(status, str):
            status = SubmissionStatus(status)
        old_status = pkg.status
        pkg.status = status
        if status == SubmissionStatus.SUBMITTED:
            pkg.submitted_at = datetime.now(timezone.utc).isoformat()
        elif status in (SubmissionStatus.CLEARED, SubmissionStatus.GRANTED, SubmissionStatus.APPROVED):
            pkg.decision_date = datetime.now(timezone.utc).isoformat()
        self._log_audit(submission_id, "status_updated", f"{old_status.value} -> {status.value}")
        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_submission(self, submission_id: str) -> SubmissionPackage | None:
        """Retrieve a submission package."""
        return self._packages.get(submission_id)

    def get_readiness_report(self, submission_id: str) -> dict[str, Any]:
        """Generate a readiness report for submission.

        Returns a summary of document statuses, AI/ML component
        readiness, and an overall readiness assessment.
        """
        pkg = self._packages.get(submission_id)
        if pkg is None:
            return {"error": "Submission not found"}

        status_counts: dict[str, int] = {}
        for doc in pkg.documents:
            s = doc.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        total = len(pkg.documents)
        finalized = status_counts.get("finalized", 0) + status_counts.get("approved", 0)
        ready = finalized == total and total > 0

        # Check AI/ML component documentation
        ai_ml_documented = all(comp.model_type != "" for comp in pkg.ai_ml_components)

        expected_timeline = _REVIEW_TIMELINES.get(pkg.submission_type, 0)

        return {
            "submission_id": submission_id,
            "submission_type": pkg.submission_type.value,
            "device_name": pkg.device_name,
            "device_class": pkg.device_class.value,
            "status": pkg.status.value,
            "total_documents": total,
            "by_status": status_counts,
            "documents_ready": finalized,
            "ready_for_submission": ready,
            "ai_ml_components": len(pkg.ai_ml_components),
            "ai_ml_documented": ai_ml_documented,
            "review_cycles": len(pkg.review_cycles),
            "expected_review_days": expected_timeline,
            "milestones": pkg.milestones,
        }

    def list_submissions(
        self,
        status: SubmissionStatus | None = None,
        submission_type: SubmissionType | None = None,
    ) -> list[SubmissionPackage]:
        """List submission packages with optional filtering."""
        pkgs = list(self._packages.values())
        if status is not None:
            pkgs = [p for p in pkgs if p.status == status]
        if submission_type is not None:
            pkgs = [p for p in pkgs if p.submission_type == submission_type]
        return pkgs

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_trail(self, submission_id: str | None = None) -> list[dict[str, Any]]:
        """Return a copy of the audit trail."""
        trail = list(self._audit_trail)
        if submission_id is not None:
            trail = [e for e in trail if e.get("submission_id") == submission_id]
        return trail

    def _log_audit(self, submission_id: str, action: str, details: str) -> None:
        """Record an audit trail entry per 21 CFR Part 11."""
        self._audit_trail.append(
            {
                "submission_id": submission_id,
                "action": action,
                "details": details,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

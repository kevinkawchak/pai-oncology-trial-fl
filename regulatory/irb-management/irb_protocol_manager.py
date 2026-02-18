"""IRB protocol lifecycle management for oncology clinical trials.

Manages Institutional Review Board protocols including initial
submissions, amendments, continuing reviews, and informed consent
versioning for AI/ML-enabled federated learning trials.

References:
    - 45 CFR Part 46: Protection of Human Subjects (Common Rule)
    - 21 CFR Parts 50, 56: FDA IRB regulations
    - ICH E6(R3): Good Clinical Practice (published September 2025)
    - OHRP Guidance on IRB Review of Research

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


class ProtocolStatus(str, Enum):
    """Lifecycle status of an IRB protocol."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    DEFERRED = "deferred"
    DISAPPROVED = "disapproved"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    CLOSED = "closed"
    EXPIRED = "expired"


class ReviewType(str, Enum):
    """Types of IRB review per 45 CFR 46.110."""

    FULL_BOARD = "full_board"
    EXPEDITED = "expedited"
    EXEMPT = "exempt"
    CONTINUING = "continuing_review"
    AMENDMENT = "amendment_review"
    REPORTABLE_EVENT = "reportable_event"


class AmendmentType(str, Enum):
    """Types of protocol amendments."""

    MINOR = "minor"
    MAJOR = "major"
    ADMINISTRATIVE = "administrative"
    SAFETY = "safety"


class ConsentType(str, Enum):
    """Types of informed consent documents."""

    FULL_CONSENT = "full_consent"
    SHORT_FORM = "short_form"
    WAIVER = "waiver_of_consent"
    ASSENT = "assent"
    RECONSENT = "reconsent"
    ADDENDUM = "consent_addendum"


class RiskLevel(str, Enum):
    """Risk classification per 45 CFR 46.102."""

    MINIMAL = "minimal_risk"
    GREATER_THAN_MINIMAL = "greater_than_minimal"
    SIGNIFICANT = "significant_risk"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConsentVersion:
    """A version of an informed consent document.

    Attributes:
        version_id: Unique version identifier.
        consent_type: Type of consent document.
        version_number: Sequential version number.
        language: Document language.
        approved_date: IRB approval date.
        effective_date: Date consent version becomes effective.
        summary_of_changes: Changes from prior version.
        ai_ml_disclosures: AI/ML-specific disclosures included.
    """

    version_id: str
    consent_type: ConsentType
    version_number: str = "1.0"
    language: str = "en"
    approved_date: str = ""
    effective_date: str = ""
    summary_of_changes: str = ""
    ai_ml_disclosures: list[str] = field(default_factory=list)


@dataclass
class ProtocolAmendment:
    """A protocol amendment submission.

    Attributes:
        amendment_id: Unique amendment identifier.
        amendment_type: Type of amendment.
        description: Description of changes.
        justification: Rationale for the amendment.
        submitted_date: When the amendment was submitted.
        approved_date: When the amendment was approved.
        status: Current status.
        affected_sections: Protocol sections affected.
        requires_reconsent: Whether participants need to reconsent.
    """

    amendment_id: str
    amendment_type: AmendmentType
    description: str = ""
    justification: str = ""
    submitted_date: str = ""
    approved_date: str = ""
    status: ProtocolStatus = ProtocolStatus.DRAFT
    affected_sections: list[str] = field(default_factory=list)
    requires_reconsent: bool = False

    def __post_init__(self) -> None:
        if not self.submitted_date:
            self.submitted_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ContinuingReview:
    """A continuing review submission.

    Attributes:
        review_id: Unique review identifier.
        review_period_start: Start of the review period.
        review_period_end: End of the review period.
        submitted_date: When the review was submitted.
        approved_date: When the review was approved.
        next_review_due: Deadline for next continuing review.
        enrollment_status: Current enrollment numbers.
        adverse_events_summary: Summary of AEs during period.
        protocol_deviations: Protocol deviations during period.
        status: Current status.
    """

    review_id: str
    review_period_start: str = ""
    review_period_end: str = ""
    submitted_date: str = ""
    approved_date: str = ""
    next_review_due: str = ""
    enrollment_status: dict[str, int] = field(default_factory=dict)
    adverse_events_summary: str = ""
    protocol_deviations: int = 0
    status: ProtocolStatus = ProtocolStatus.SUBMITTED


@dataclass
class IRBProtocol:
    """A complete IRB protocol for a clinical trial.

    Attributes:
        protocol_id: Unique protocol identifier.
        title: Full protocol title.
        short_title: Abbreviated title.
        version: Protocol version.
        status: Current lifecycle status.
        review_type: Type of IRB review.
        risk_level: Risk classification.
        pi_name: Principal Investigator name.
        pi_institution: PI institution.
        irb_name: Name of the reviewing IRB.
        irb_number: IRB registration number.
        sponsor: Study sponsor.
        study_population: Target study population description.
        enrollment_target: Target enrollment number.
        current_enrollment: Current enrollment count.
        sites: Participating sites.
        consent_versions: Informed consent version history.
        amendments: Protocol amendments.
        continuing_reviews: Continuing review history.
        ai_ml_components: AI/ML components described in protocol.
        submission_date: Initial IRB submission date.
        approval_date: Initial IRB approval date.
        expiration_date: Current approval expiration date.
        created_at: Protocol creation timestamp.
        audit_trail: Protocol action history.
    """

    protocol_id: str
    title: str
    short_title: str = ""
    version: str = "1.0"
    status: ProtocolStatus = ProtocolStatus.DRAFT
    review_type: ReviewType = ReviewType.FULL_BOARD
    risk_level: RiskLevel = RiskLevel.MINIMAL
    pi_name: str = ""
    pi_institution: str = ""
    irb_name: str = ""
    irb_number: str = ""
    sponsor: str = ""
    study_population: str = ""
    enrollment_target: int = 0
    current_enrollment: int = 0
    sites: list[str] = field(default_factory=list)
    consent_versions: list[ConsentVersion] = field(default_factory=list)
    amendments: list[ProtocolAmendment] = field(default_factory=list)
    continuing_reviews: list[ContinuingReview] = field(default_factory=list)
    ai_ml_components: list[str] = field(default_factory=list)
    submission_date: str = ""
    approval_date: str = ""
    expiration_date: str = ""
    created_at: str = ""
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_active(self) -> bool:
        """Whether the protocol is currently active (approved and not expired)."""
        return self.status in (
            ProtocolStatus.APPROVED,
            ProtocolStatus.APPROVED_WITH_CONDITIONS,
        )

    @property
    def days_until_expiration(self) -> int | None:
        """Days remaining until the protocol expires."""
        if not self.expiration_date:
            return None
        try:
            exp = datetime.fromisoformat(self.expiration_date)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            delta = exp - datetime.now(timezone.utc)
            return max(0, delta.days)
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# IRB Protocol Manager
# ---------------------------------------------------------------------------


class IRBProtocolManager:
    """Manages IRB protocol lifecycle for oncology clinical trials.

    Handles initial submissions, amendments, continuing reviews,
    consent versioning, and compliance tracking per 45 CFR Part 46,
    21 CFR Parts 50/56, and ICH E6(R3) published September 2025.
    """

    def __init__(self) -> None:
        self._protocols: dict[str, IRBProtocol] = {}
        self._audit_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Protocol management
    # ------------------------------------------------------------------

    def create_protocol(
        self,
        protocol_id: str,
        title: str,
        pi_name: str = "",
        pi_institution: str = "",
        review_type: ReviewType | str = ReviewType.FULL_BOARD,
        risk_level: RiskLevel | str = RiskLevel.MINIMAL,
        enrollment_target: int = 0,
        sites: list[str] | None = None,
    ) -> IRBProtocol:
        """Create a new IRB protocol.

        Args:
            protocol_id: Unique protocol identifier.
            title: Full protocol title.
            pi_name: Principal Investigator name.
            pi_institution: PI institution.
            review_type: Type of IRB review.
            risk_level: Risk classification.
            enrollment_target: Target enrollment.
            sites: Participating sites.

        Returns:
            The created IRBProtocol.
        """
        if isinstance(review_type, str):
            review_type = ReviewType(review_type)
        if isinstance(risk_level, str):
            risk_level = RiskLevel(risk_level)

        protocol = IRBProtocol(
            protocol_id=protocol_id,
            title=title,
            pi_name=pi_name,
            pi_institution=pi_institution,
            review_type=review_type,
            risk_level=risk_level,
            enrollment_target=enrollment_target,
            sites=list(sites or []),
        )
        self._protocols[protocol_id] = protocol
        self._log_action(protocol_id, "protocol_created", f"title={title}")
        logger.info("IRB protocol '%s' created", protocol_id)
        return protocol

    def submit_protocol(self, protocol_id: str, irb_name: str = "", irb_number: str = "") -> bool:
        """Submit a protocol for IRB review."""
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return False
        protocol.status = ProtocolStatus.SUBMITTED
        protocol.submission_date = datetime.now(timezone.utc).isoformat()
        if irb_name:
            protocol.irb_name = irb_name
        if irb_number:
            protocol.irb_number = irb_number
        self._log_action(protocol_id, "protocol_submitted", f"irb={irb_name}")
        return True

    def approve_protocol(
        self,
        protocol_id: str,
        approval_period_days: int = 365,
        conditions: str = "",
    ) -> bool:
        """Record IRB approval of a protocol.

        Args:
            protocol_id: Protocol to approve.
            approval_period_days: Length of approval period.
            conditions: Any conditions of approval.

        Returns:
            True if the protocol was approved.
        """
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return False

        now = datetime.now(timezone.utc)
        protocol.approval_date = now.isoformat()
        protocol.expiration_date = (now + timedelta(days=approval_period_days)).isoformat()

        if conditions:
            protocol.status = ProtocolStatus.APPROVED_WITH_CONDITIONS
        else:
            protocol.status = ProtocolStatus.APPROVED

        self._log_action(
            protocol_id,
            "protocol_approved",
            f"period={approval_period_days}d, conditions={bool(conditions)}",
        )
        logger.info("IRB protocol '%s' approved for %d days", protocol_id, approval_period_days)
        return True

    def update_status(self, protocol_id: str, status: ProtocolStatus | str, reason: str = "") -> bool:
        """Update the protocol status."""
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return False
        if isinstance(status, str):
            status = ProtocolStatus(status)
        old_status = protocol.status
        protocol.status = status
        self._log_action(protocol_id, "status_changed", f"{old_status.value} -> {status.value}: {reason}")
        return True

    # ------------------------------------------------------------------
    # Amendment management
    # ------------------------------------------------------------------

    def submit_amendment(
        self,
        protocol_id: str,
        amendment_type: AmendmentType | str,
        description: str,
        justification: str = "",
        affected_sections: list[str] | None = None,
        requires_reconsent: bool = False,
    ) -> str | None:
        """Submit a protocol amendment.

        Args:
            protocol_id: Protocol to amend.
            amendment_type: Type of amendment.
            description: Description of changes.
            justification: Rationale for the amendment.
            affected_sections: Protocol sections affected.
            requires_reconsent: Whether participants need reconsent.

        Returns:
            The amendment ID, or None if protocol not found.
        """
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return None
        if isinstance(amendment_type, str):
            amendment_type = AmendmentType(amendment_type)

        amendment_id = f"{protocol_id}-AMD-{len(protocol.amendments) + 1:02d}"
        amendment = ProtocolAmendment(
            amendment_id=amendment_id,
            amendment_type=amendment_type,
            description=description,
            justification=justification,
            affected_sections=list(affected_sections or []),
            requires_reconsent=requires_reconsent,
            status=ProtocolStatus.SUBMITTED,
        )
        protocol.amendments.append(amendment)
        self._log_action(protocol_id, "amendment_submitted", f"id={amendment_id}, type={amendment_type.value}")
        return amendment_id

    def approve_amendment(self, protocol_id: str, amendment_id: str) -> bool:
        """Approve a protocol amendment."""
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return False

        for amendment in protocol.amendments:
            if amendment.amendment_id == amendment_id:
                amendment.status = ProtocolStatus.APPROVED
                amendment.approved_date = datetime.now(timezone.utc).isoformat()
                # Increment protocol version
                try:
                    major, minor = protocol.version.split(".")
                    if amendment.amendment_type in (AmendmentType.MAJOR, AmendmentType.SAFETY):
                        protocol.version = f"{int(major) + 1}.0"
                    else:
                        protocol.version = f"{major}.{int(minor) + 1}"
                except (ValueError, AttributeError):
                    pass
                self._log_action(protocol_id, "amendment_approved", f"id={amendment_id}")
                return True
        return False

    # ------------------------------------------------------------------
    # Consent versioning
    # ------------------------------------------------------------------

    def add_consent_version(
        self,
        protocol_id: str,
        consent_type: ConsentType | str,
        version_number: str = "",
        language: str = "en",
        changes: str = "",
        ai_ml_disclosures: list[str] | None = None,
    ) -> str | None:
        """Add a new consent version to a protocol.

        Args:
            protocol_id: Target protocol.
            consent_type: Type of consent document.
            version_number: Version number (auto-incremented if empty).
            language: Document language code.
            changes: Summary of changes from prior version.
            ai_ml_disclosures: AI/ML-specific disclosures.

        Returns:
            The consent version ID, or None if protocol not found.
        """
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return None
        if isinstance(consent_type, str):
            consent_type = ConsentType(consent_type)

        if not version_number:
            existing = [cv for cv in protocol.consent_versions if cv.consent_type == consent_type]
            version_number = f"{len(existing) + 1}.0"

        version_id = f"{protocol_id}-ICF-{len(protocol.consent_versions) + 1:02d}"
        consent = ConsentVersion(
            version_id=version_id,
            consent_type=consent_type,
            version_number=version_number,
            language=language,
            summary_of_changes=changes,
            ai_ml_disclosures=list(ai_ml_disclosures or []),
        )
        protocol.consent_versions.append(consent)
        self._log_action(protocol_id, "consent_version_added", f"id={version_id}, v={version_number}")
        return version_id

    def approve_consent(self, protocol_id: str, version_id: str) -> bool:
        """Approve a consent version."""
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return False

        now = datetime.now(timezone.utc).isoformat()
        for cv in protocol.consent_versions:
            if cv.version_id == version_id:
                cv.approved_date = now
                cv.effective_date = now
                self._log_action(protocol_id, "consent_approved", f"id={version_id}")
                return True
        return False

    # ------------------------------------------------------------------
    # Continuing review
    # ------------------------------------------------------------------

    def submit_continuing_review(
        self,
        protocol_id: str,
        enrollment_status: dict[str, int] | None = None,
        adverse_events_summary: str = "",
        protocol_deviations: int = 0,
    ) -> str | None:
        """Submit a continuing review.

        Args:
            protocol_id: Target protocol.
            enrollment_status: Current enrollment numbers.
            adverse_events_summary: Summary of AEs during review period.
            protocol_deviations: Number of protocol deviations.

        Returns:
            The review ID, or None if protocol not found.
        """
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return None

        review_id = f"{protocol_id}-CR-{len(protocol.continuing_reviews) + 1:02d}"
        review = ContinuingReview(
            review_id=review_id,
            review_period_start=protocol.approval_date,
            review_period_end=datetime.now(timezone.utc).isoformat(),
            enrollment_status=dict(enrollment_status or {}),
            adverse_events_summary=adverse_events_summary,
            protocol_deviations=protocol_deviations,
        )
        protocol.continuing_reviews.append(review)
        self._log_action(protocol_id, "continuing_review_submitted", f"id={review_id}")
        return review_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_protocol(self, protocol_id: str) -> IRBProtocol | None:
        """Retrieve a protocol by ID."""
        return self._protocols.get(protocol_id)

    def get_expiring_protocols(self, days: int = 90) -> list[IRBProtocol]:
        """Return protocols expiring within the specified number of days."""
        expiring: list[IRBProtocol] = []
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days)

        for protocol in self._protocols.values():
            if not protocol.is_active or not protocol.expiration_date:
                continue
            try:
                exp = datetime.fromisoformat(protocol.expiration_date)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp <= cutoff:
                    expiring.append(protocol)
            except (ValueError, TypeError):
                continue

        return expiring

    def list_protocols(self, status: ProtocolStatus | None = None) -> list[IRBProtocol]:
        """List protocols with optional status filter."""
        protocols = list(self._protocols.values())
        if status is not None:
            protocols = [p for p in protocols if p.status == status]
        return protocols

    def get_compliance_summary(self, protocol_id: str) -> dict[str, Any]:
        """Generate a compliance summary for a protocol."""
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            return {"error": "Protocol not found"}

        return {
            "protocol_id": protocol_id,
            "status": protocol.status.value,
            "version": protocol.version,
            "is_active": protocol.is_active,
            "days_until_expiration": protocol.days_until_expiration,
            "enrollment": f"{protocol.current_enrollment}/{protocol.enrollment_target}",
            "amendments": len(protocol.amendments),
            "consent_versions": len(protocol.consent_versions),
            "continuing_reviews": len(protocol.continuing_reviews),
            "sites": len(protocol.sites),
            "audit_trail_entries": len(protocol.audit_trail),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_action(self, protocol_id: str, action: str, details: str) -> None:
        """Record an audit trail entry."""
        entry = {
            "protocol_id": protocol_id,
            "action": action,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        protocol = self._protocols.get(protocol_id)
        if protocol is not None:
            protocol.audit_trail.append(entry)
        self._audit_log.append(entry)

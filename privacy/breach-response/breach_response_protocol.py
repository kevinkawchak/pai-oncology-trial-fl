"""Breach detection, risk assessment, and response protocol.

Implements a structured incident response workflow for PHI/PII
breaches compliant with HIPAA Breach Notification Rule
(45 CFR 164.400-414) and GDPR Article 33-34 notification requirements.

Includes risk assessment per the four-factor test defined in
45 CFR 164.402, 60-day notification timeline tracking, and
remediation workflow management.

References:
    - 45 CFR 164.400-414: HIPAA Breach Notification Rule
    - 45 CFR 164.402: Four-factor risk assessment
    - GDPR Articles 33-34: Breach notification
    - HHS Breach Notification Guidance (2009, updated 2013)
    - NIST SP 800-61r2: Computer Security Incident Handling Guide

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BreachSeverity(str, Enum):
    """Breach severity classification.

    Aligned with NIST SP 800-61r2 severity categories and
    HIPAA risk assessment outcomes.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentPhase(str, Enum):
    """Incident response lifecycle phases per NIST SP 800-61r2."""

    DETECTED = "detected"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    NOTIFIED = "notified"
    CLOSED = "closed"


class NotificationType(str, Enum):
    """Types of regulatory notifications required."""

    HHS_OCR = "hhs_ocr"  # HIPAA — HHS Office for Civil Rights
    STATE_AG = "state_attorney_general"  # State breach notification
    INDIVIDUAL = "individual_notification"  # Affected individuals
    MEDIA = "media_notification"  # Required for 500+ individuals
    GDPR_DPA = "gdpr_data_protection_authority"  # GDPR supervisory authority
    GDPR_INDIVIDUAL = "gdpr_individual_notification"  # GDPR data subject


class RemediationStatus(str, Enum):
    """Status of a remediation action."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RiskAssessment:
    """Four-factor risk assessment per 45 CFR 164.402.

    The four factors are:
    1. Nature and extent of PHI involved
    2. Unauthorized person who used/accessed the PHI
    3. Whether PHI was actually acquired or viewed
    4. Extent to which risk has been mitigated

    Each factor is scored 0.0-1.0, and the composite risk score
    determines whether breach notification is required.

    Attributes:
        phi_nature_score: Factor 1 — nature/extent of PHI (0.0-1.0).
        unauthorized_access_score: Factor 2 — who accessed (0.0-1.0).
        acquisition_score: Factor 3 — was PHI acquired/viewed (0.0-1.0).
        mitigation_score: Factor 4 — mitigation extent (0.0-1.0, higher=less mitigated).
        notes: Assessment justification.
    """

    phi_nature_score: float = 0.0
    unauthorized_access_score: float = 0.0
    acquisition_score: float = 0.0
    mitigation_score: float = 0.0
    notes: str = ""

    def calculate_risk(self) -> float:
        """Calculate composite risk score (0.0-1.0).

        Weighted average of the four factors with clamping
        to ensure the score stays within [0.0, 1.0] even
        if individual factors are out of range.
        """
        # Clamp individual scores to [0.0, 1.0]
        f1 = max(0.0, min(1.0, self.phi_nature_score))
        f2 = max(0.0, min(1.0, self.unauthorized_access_score))
        f3 = max(0.0, min(1.0, self.acquisition_score))
        f4 = max(0.0, min(1.0, self.mitigation_score))

        # Weighted composite — factor 1 and 3 weighted higher
        raw = 0.30 * f1 + 0.20 * f2 + 0.30 * f3 + 0.20 * f4
        return max(0.0, min(1.0, raw))

    @property
    def requires_notification(self) -> bool:
        """Whether the risk level triggers notification requirements.

        Per HIPAA, notification is required unless the covered entity
        demonstrates a low probability of compromise. A composite
        score >= 0.5 indicates notification is likely required.
        """
        return self.calculate_risk() >= 0.5


@dataclass
class BreachNotification:
    """A regulatory notification for a breach incident.

    Attributes:
        notification_type: Type of notification.
        due_date: Deadline for the notification (ISO format).
        sent_date: When the notification was actually sent.
        recipient: Who was notified.
        status: Current status.
        content_summary: Summary of notification content (no PHI).
    """

    notification_type: NotificationType
    due_date: str = ""
    sent_date: str = ""
    recipient: str = ""
    status: RemediationStatus = RemediationStatus.PLANNED
    content_summary: str = ""


@dataclass
class RemediationAction:
    """A remediation action for a breach incident.

    Attributes:
        action_id: Unique action identifier.
        description: What needs to be done.
        assigned_to: Person or team responsible.
        status: Current status.
        created_at: When the action was created.
        completed_at: When the action was completed.
        verification_notes: Notes from verification step.
    """

    action_id: str
    description: str
    assigned_to: str = ""
    status: RemediationStatus = RemediationStatus.PLANNED
    created_at: str = ""
    completed_at: str = ""
    verification_notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class BreachIncident:
    """A tracked breach incident with full response lifecycle.

    Attributes:
        incident_id: Unique incident identifier.
        severity: Overall severity assessment.
        phase: Current lifecycle phase.
        description: Incident description (no PHI).
        risk_assessment: Four-factor risk assessment.
        detected_at: When the breach was detected.
        contained_at: When containment was achieved.
        resolved_at: When the incident was resolved.
        notifications: Required and sent notifications.
        remediation_actions: Ordered remediation steps.
        affected_sites: List of site IDs affected.
        affected_individuals: Estimated number of affected individuals.
        phi_types_involved: Types of PHI involved.
        root_cause: Determined root cause.
        lessons_learned: Post-incident findings.
        audit_trail: Timestamped log of all actions taken.
    """

    incident_id: str
    severity: BreachSeverity
    phase: IncidentPhase = IncidentPhase.DETECTED
    description: str = ""
    risk_assessment: RiskAssessment | None = None
    detected_at: str = ""
    contained_at: str = ""
    resolved_at: str = ""
    notifications: list[BreachNotification] = field(default_factory=list)
    remediation_actions: list[RemediationAction] = field(default_factory=list)
    affected_sites: list[str] = field(default_factory=list)
    affected_individuals: int = 0
    phi_types_involved: list[str] = field(default_factory=list)
    root_cause: str = ""
    lessons_learned: str = ""
    audit_trail: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()

    @property
    def days_since_detection(self) -> int:
        """Days elapsed since breach detection."""
        try:
            detected = datetime.fromisoformat(self.detected_at)
            if detected.tzinfo is None:
                detected = detected.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - detected
            return max(0, delta.days)
        except (ValueError, TypeError):
            return 0

    @property
    def notification_deadline(self) -> str:
        """60-day HIPAA notification deadline (ISO format).

        Per 45 CFR 164.404(b), notification must be provided
        without unreasonable delay, no later than 60 days from
        the date the breach was discovered.
        """
        try:
            detected = datetime.fromisoformat(self.detected_at)
            if detected.tzinfo is None:
                detected = detected.replace(tzinfo=timezone.utc)
            deadline = detected + timedelta(days=60)
            return deadline.isoformat()
        except (ValueError, TypeError):
            return ""


# ---------------------------------------------------------------------------
# Breach Response Protocol
# ---------------------------------------------------------------------------


class BreachResponseProtocol:
    """Manages breach detection, risk assessment, and response workflow.

    Implements the HIPAA Breach Notification Rule (45 CFR 164.400-414)
    including the four-factor risk assessment, 60-day notification
    timeline, and structured remediation tracking.

    Args:
        organization_name: Name of the covered entity.
        hipaa_privacy_officer: Contact for the HIPAA privacy officer.
    """

    def __init__(
        self,
        organization_name: str = "PAI Oncology Trial Consortium",
        hipaa_privacy_officer: str = "",
    ) -> None:
        self.organization_name = organization_name
        self.hipaa_privacy_officer = hipaa_privacy_officer
        self._incidents: dict[str, BreachIncident] = {}
        self._next_incident_id: int = 1

    # ------------------------------------------------------------------
    # Incident creation
    # ------------------------------------------------------------------

    def create_incident(
        self,
        severity: BreachSeverity,
        description: str = "",
        affected_sites: list[str] | None = None,
        affected_individuals: int = 0,
        phi_types: list[str] | None = None,
    ) -> BreachIncident:
        """Create a new breach incident.

        Args:
            severity: Assessed severity.
            description: Incident description (must not contain PHI).
            affected_sites: List of affected site IDs.
            affected_individuals: Estimated count of affected individuals.
            phi_types: Types of PHI involved.

        Returns:
            The created BreachIncident.
        """
        iid = f"BRI-{self._next_incident_id:04d}"
        self._next_incident_id += 1

        incident = BreachIncident(
            incident_id=iid,
            severity=severity,
            description=description,
            affected_sites=list(affected_sites or []),
            affected_individuals=affected_individuals,
            phi_types_involved=list(phi_types or []),
        )

        # Auto-generate notification requirements
        incident.notifications = self._generate_notifications(incident)

        self._incidents[iid] = incident
        self._log_action(incident, "incident_created", f"severity={severity.value}")

        logger.warning(
            "Breach incident %s created: severity=%s, affected=%d individuals",
            iid,
            severity.value,
            affected_individuals,
        )
        return incident

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def triage(self, incident_id: str, risk_assessment: RiskAssessment) -> bool:
        """Perform initial triage with risk assessment."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.phase = IncidentPhase.TRIAGED
        inc.risk_assessment = risk_assessment
        self._log_action(inc, "triaged", f"risk_score={risk_assessment.calculate_risk():.2f}")
        return True

    def investigate(self, incident_id: str, notes: str = "") -> bool:
        """Move incident to investigation phase."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.phase = IncidentPhase.INVESTIGATING
        self._log_action(inc, "investigation_started", notes)
        return True

    def contain(self, incident_id: str, actions_taken: str = "") -> bool:
        """Mark incident as contained."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.phase = IncidentPhase.CONTAINED
        inc.contained_at = datetime.now(timezone.utc).isoformat()
        self._log_action(inc, "contained", actions_taken)
        return True

    def resolve(self, incident_id: str, root_cause: str = "", lessons: str = "") -> bool:
        """Mark incident as resolved."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.phase = IncidentPhase.RESOLVED
        inc.resolved_at = datetime.now(timezone.utc).isoformat()
        inc.root_cause = root_cause
        inc.lessons_learned = lessons
        self._log_action(inc, "resolved", f"root_cause={root_cause}")
        return True

    def send_notification(
        self,
        incident_id: str,
        notification_type: NotificationType,
        recipient: str = "",
    ) -> bool:
        """Record that a notification was sent."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False

        for notif in inc.notifications:
            if notif.notification_type == notification_type:
                notif.sent_date = datetime.now(timezone.utc).isoformat()
                notif.status = RemediationStatus.COMPLETED
                notif.recipient = recipient
                self._log_action(
                    inc,
                    "notification_sent",
                    f"type={notification_type.value}, recipient={recipient}",
                )
                return True

        return False

    def close_incident(self, incident_id: str) -> bool:
        """Close an incident after all notifications and remediation."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.phase = IncidentPhase.CLOSED
        self._log_action(inc, "incident_closed", "")
        logger.info("Breach incident %s closed", incident_id)
        return True

    # ------------------------------------------------------------------
    # Remediation
    # ------------------------------------------------------------------

    def add_remediation(
        self,
        incident_id: str,
        description: str,
        assigned_to: str = "",
    ) -> str | None:
        """Add a remediation action to an incident.

        Returns:
            The remediation action ID, or None if the incident was not found.
        """
        inc = self._incidents.get(incident_id)
        if inc is None:
            return None

        action_id = f"{incident_id}-REM-{len(inc.remediation_actions) + 1:02d}"
        action = RemediationAction(
            action_id=action_id,
            description=description,
            assigned_to=assigned_to,
        )
        inc.remediation_actions.append(action)
        self._log_action(inc, "remediation_added", f"action={action_id}")
        return action_id

    def complete_remediation(
        self,
        incident_id: str,
        action_id: str,
        verification_notes: str = "",
    ) -> bool:
        """Mark a remediation action as completed."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False

        for action in inc.remediation_actions:
            if action.action_id == action_id:
                action.status = RemediationStatus.COMPLETED
                action.completed_at = datetime.now(timezone.utc).isoformat()
                action.verification_notes = verification_notes
                self._log_action(inc, "remediation_completed", f"action={action_id}")
                return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_incident(self, incident_id: str) -> BreachIncident | None:
        """Retrieve an incident by ID."""
        return self._incidents.get(incident_id)

    def get_open_incidents(self) -> list[BreachIncident]:
        """Return all incidents not yet closed."""
        return [i for i in self._incidents.values() if i.phase != IncidentPhase.CLOSED]

    def get_overdue_notifications(self) -> list[tuple[str, BreachNotification]]:
        """Return notifications that are past their due date.

        Checks against the 60-day HIPAA notification deadline.
        """
        overdue: list[tuple[str, BreachNotification]] = []
        now = datetime.now(timezone.utc)

        for iid, inc in self._incidents.items():
            for notif in inc.notifications:
                if notif.status == RemediationStatus.COMPLETED:
                    continue
                if notif.due_date:
                    try:
                        due = datetime.fromisoformat(notif.due_date)
                        if due.tzinfo is None:
                            due = due.replace(tzinfo=timezone.utc)
                        if now > due:
                            overdue.append((iid, notif))
                    except (ValueError, TypeError):
                        continue

        return overdue

    def get_timeline_report(self, incident_id: str) -> dict:
        """Generate a timeline report for an incident."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return {"error": "Incident not found"}

        return {
            "incident_id": incident_id,
            "severity": inc.severity.value,
            "phase": inc.phase.value,
            "detected_at": inc.detected_at,
            "contained_at": inc.contained_at,
            "resolved_at": inc.resolved_at,
            "days_since_detection": inc.days_since_detection,
            "notification_deadline": inc.notification_deadline,
            "affected_individuals": inc.affected_individuals,
            "notifications_sent": sum(1 for n in inc.notifications if n.status == RemediationStatus.COMPLETED),
            "notifications_pending": sum(1 for n in inc.notifications if n.status != RemediationStatus.COMPLETED),
            "remediations_completed": sum(
                1 for a in inc.remediation_actions if a.status == RemediationStatus.COMPLETED
            ),
            "remediations_pending": sum(1 for a in inc.remediation_actions if a.status != RemediationStatus.COMPLETED),
            "audit_trail_entries": len(inc.audit_trail),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_notifications(self, incident: BreachIncident) -> list[BreachNotification]:
        """Generate required notifications based on incident parameters."""
        notifications: list[BreachNotification] = []
        deadline_60 = ""

        try:
            detected = datetime.fromisoformat(incident.detected_at)
            if detected.tzinfo is None:
                detected = detected.replace(tzinfo=timezone.utc)
            deadline_60 = (detected + timedelta(days=60)).isoformat()
        except (ValueError, TypeError):
            pass

        # HIPAA — HHS OCR notification (always required for confirmed breaches)
        notifications.append(
            BreachNotification(
                notification_type=NotificationType.HHS_OCR,
                due_date=deadline_60,
                content_summary="HIPAA breach notification to HHS OCR",
            )
        )

        # Individual notification (always required)
        notifications.append(
            BreachNotification(
                notification_type=NotificationType.INDIVIDUAL,
                due_date=deadline_60,
                content_summary="Individual breach notification letters",
            )
        )

        # Media notification (required for 500+ individuals in a state)
        if incident.affected_individuals >= 500:
            notifications.append(
                BreachNotification(
                    notification_type=NotificationType.MEDIA,
                    due_date=deadline_60,
                    content_summary="Media notification for large-scale breach",
                )
            )

        # State AG notification
        notifications.append(
            BreachNotification(
                notification_type=NotificationType.STATE_AG,
                due_date=deadline_60,
                content_summary="State attorney general breach notification",
            )
        )

        return notifications

    @staticmethod
    def _log_action(incident: BreachIncident, action: str, details: str) -> None:
        """Append an audit trail entry to the incident."""
        incident.audit_trail.append(
            {
                "action": action,
                "details": details,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": incident.phase.value,
            }
        )

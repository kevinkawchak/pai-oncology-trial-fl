"""Breach detection and response protocol for federated learning.

Provides automated detection of potential data breaches and a
structured incident response workflow compliant with HIPAA (72-hour
notification) and GDPR (72-hour supervisory authority notification)
requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class BreachSeverity(str, Enum):
    """Severity classification for a detected breach."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Lifecycle status of a breach incident."""

    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    REPORTED = "reported"


@dataclass
class BreachIndicator:
    """A single indicator that may signal a data breach.

    Attributes:
        indicator_type: Category (e.g. ``"anomalous_access"``,
            ``"phi_in_transit"``, ``"failed_auth"``).
        description: Human-readable description.
        severity: Assessed severity level.
        source: System or module that generated the indicator.
        timestamp: When the indicator was detected.
        metadata: Additional context.
    """

    indicator_type: str
    description: str
    severity: BreachSeverity = BreachSeverity.LOW
    source: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class BreachIncident:
    """A tracked breach incident with response actions.

    Attributes:
        incident_id: Unique incident identifier.
        severity: Overall severity of the incident.
        status: Current lifecycle status.
        indicators: Breach indicators associated with this incident.
        detected_at: When the incident was first detected.
        contained_at: When containment was achieved.
        resolved_at: When the incident was resolved.
        reported_at: When regulatory authorities were notified.
        response_actions: Ordered list of response actions taken.
        affected_sites: List of site IDs affected.
        affected_patients: Estimated number of affected patients.
        root_cause: Determined root cause (once investigated).
    """

    incident_id: str
    severity: BreachSeverity
    status: IncidentStatus = IncidentStatus.DETECTED
    indicators: list[BreachIndicator] = field(default_factory=list)
    detected_at: str = ""
    contained_at: str | None = None
    resolved_at: str | None = None
    reported_at: str | None = None
    response_actions: list[dict] = field(default_factory=list)
    affected_sites: list[str] = field(default_factory=list)
    affected_patients: int = 0
    root_cause: str = ""

    def __post_init__(self) -> None:
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()


class BreachResponseProtocol:
    """Automated breach detection and incident response.

    Monitors for breach indicators, creates incidents, and tracks
    the response workflow through detection, investigation,
    containment, resolution, and regulatory reporting.

    Args:
        auto_escalate_threshold: Number of indicators within the
            escalation window that triggers automatic incident creation.
        escalation_window_minutes: Time window for indicator counting.
    """

    def __init__(
        self,
        auto_escalate_threshold: int = 3,
        escalation_window_minutes: int = 60,
    ):
        self.auto_escalate_threshold = auto_escalate_threshold
        self.escalation_window_minutes = escalation_window_minutes
        self._indicators: list[BreachIndicator] = []
        self._incidents: dict[str, BreachIncident] = {}
        self._next_incident_id = 1

    # ------------------------------------------------------------------
    # Indicator reporting
    # ------------------------------------------------------------------

    def report_indicator(self, indicator: BreachIndicator) -> str | None:
        """Report a breach indicator.

        If the number of recent indicators exceeds the escalation
        threshold, an incident is automatically created.

        Args:
            indicator: The detected breach indicator.

        Returns:
            Incident ID if auto-escalation triggered, else None.
        """
        self._indicators.append(indicator)
        logger.warning(
            "Breach indicator: %s — %s (severity=%s)",
            indicator.indicator_type,
            indicator.description,
            indicator.severity.value,
        )

        # Check for auto-escalation
        recent = self._recent_indicators()
        if len(recent) >= self.auto_escalate_threshold:
            return self.create_incident(
                severity=self._highest_severity(recent),
                indicators=recent,
            )
        return None

    # ------------------------------------------------------------------
    # Incident lifecycle
    # ------------------------------------------------------------------

    def create_incident(
        self,
        severity: BreachSeverity = BreachSeverity.MEDIUM,
        indicators: list[BreachIndicator] | None = None,
        affected_sites: list[str] | None = None,
    ) -> str:
        """Create a new breach incident.

        Args:
            severity: Overall severity assessment.
            indicators: Associated breach indicators.
            affected_sites: Site IDs involved.

        Returns:
            The incident ID.
        """
        iid = f"INC-{self._next_incident_id:04d}"
        self._next_incident_id += 1

        incident = BreachIncident(
            incident_id=iid,
            severity=severity,
            indicators=list(indicators or []),
            affected_sites=list(affected_sites or []),
        )
        self._incidents[iid] = incident
        logger.warning("Breach incident created: %s (severity=%s)", iid, severity.value)
        return iid

    def investigate(self, incident_id: str, notes: str = "") -> bool:
        """Move an incident to the investigating state."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.status = IncidentStatus.INVESTIGATING
        inc.response_actions.append(self._action("investigation_started", notes))
        return True

    def contain(self, incident_id: str, actions_taken: str = "") -> bool:
        """Mark an incident as contained."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.status = IncidentStatus.CONTAINED
        inc.contained_at = datetime.now(timezone.utc).isoformat()
        inc.response_actions.append(self._action("contained", actions_taken))
        return True

    def resolve(self, incident_id: str, root_cause: str = "", notes: str = "") -> bool:
        """Mark an incident as resolved."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.status = IncidentStatus.RESOLVED
        inc.resolved_at = datetime.now(timezone.utc).isoformat()
        inc.root_cause = root_cause
        inc.response_actions.append(self._action("resolved", notes))
        return True

    def report_to_authorities(self, incident_id: str, authority: str = "HHS_OCR") -> bool:
        """Record that the incident was reported to a regulatory authority.

        HIPAA requires reporting to the HHS Office for Civil Rights
        within 60 days (or 72 hours for breaches affecting 500+
        individuals).  GDPR requires notification to the supervisory
        authority within 72 hours.
        """
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.status = IncidentStatus.REPORTED
        inc.reported_at = datetime.now(timezone.utc).isoformat()
        inc.response_actions.append(self._action("reported_to_authority", authority))
        logger.info("Incident %s reported to %s", incident_id, authority)
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_incident(self, incident_id: str) -> BreachIncident | None:
        """Retrieve an incident by ID."""
        return self._incidents.get(incident_id)

    def get_open_incidents(self) -> list[BreachIncident]:
        """Return all incidents that are not yet resolved/reported."""
        closed = {IncidentStatus.RESOLVED, IncidentStatus.REPORTED}
        return [i for i in self._incidents.values() if i.status not in closed]

    def get_all_indicators(self) -> list[BreachIndicator]:
        """Return all reported indicators."""
        return list(self._indicators)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recent_indicators(self) -> list[BreachIndicator]:
        """Return indicators within the escalation window."""
        now = datetime.now(timezone.utc)
        recent: list[BreachIndicator] = []
        for ind in self._indicators:
            try:
                ts = datetime.fromisoformat(ind.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                diff = (now - ts).total_seconds() / 60.0
                if diff <= self.escalation_window_minutes:
                    recent.append(ind)
            except (ValueError, TypeError):
                recent.append(ind)
        return recent

    @staticmethod
    def _highest_severity(indicators: list[BreachIndicator]) -> BreachSeverity:
        order = [
            BreachSeverity.LOW,
            BreachSeverity.MEDIUM,
            BreachSeverity.HIGH,
            BreachSeverity.CRITICAL,
        ]
        highest = BreachSeverity.LOW
        for ind in indicators:
            if order.index(ind.severity) > order.index(highest):
                highest = ind.severity
        return highest

    @staticmethod
    def _action(action_type: str, details: str = "") -> dict:
        return {
            "action": action_type,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

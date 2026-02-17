"""HIPAA-compliant audit logging for federated learning operations.

Records all data access, model training events, and potential
security incidents to maintain an immutable audit trail for
regulatory compliance.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Categories of auditable events."""

    DATA_ACCESS = "data_access"
    MODEL_TRAINING = "model_training"
    MODEL_AGGREGATION = "model_aggregation"
    PARAMETER_EXCHANGE = "parameter_exchange"
    CONSENT_CHANGE = "consent_change"
    PHI_DETECTED = "phi_detected"
    DEIDENTIFICATION = "deidentification"
    BREACH_DETECTED = "breach_detected"
    SYSTEM_EVENT = "system_event"


class Severity(str, Enum):
    """Event severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """A single auditable event.

    Attributes:
        event_type: Category of the event.
        severity: Severity level.
        actor: Identity of the user/system that triggered the event.
        resource: The resource that was accessed or modified.
        action: Description of the action performed.
        details: Additional event-specific information.
        timestamp: ISO-format timestamp.
        integrity_hash: SHA-256 hash for tamper detection.
    """

    event_type: EventType
    severity: Severity
    actor: str
    resource: str
    action: str
    details: dict = field(default_factory=dict)
    timestamp: str = ""
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash for tamper detection."""
        payload = (
            f"{self.event_type.value}:{self.actor}:{self.resource}:{self.action}:{self.timestamp}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "timestamp": self.timestamp,
            "integrity_hash": self.integrity_hash,
        }


class AuditLogger:
    """HIPAA-compliant audit logger for federated learning.

    Maintains an append-only event log with integrity hashing.
    Supports file-based persistence and breach detection alerts.

    Args:
        log_path: Optional file path for persistent logging.
            If None, events are stored in memory only.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: EventType,
        actor: str,
        resource: str,
        action: str,
        severity: Severity = Severity.INFO,
        details: dict | None = None,
    ) -> AuditEvent:
        """Record an audit event.

        Args:
            event_type: Category of the event.
            actor: Who initiated the event.
            resource: What was accessed or modified.
            action: Description of the action.
            severity: Event severity.
            details: Extra metadata.

        Returns:
            The created AuditEvent.
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            actor=actor,
            resource=resource,
            action=action,
            details=details or {},
        )
        self._events.append(event)

        if self._log_path:
            self._persist_event(event)

        if severity in (Severity.ERROR, Severity.CRITICAL):
            logger.warning(
                "AUDIT [%s] %s: %s -> %s (%s)",
                severity.value.upper(),
                event_type.value,
                actor,
                resource,
                action,
            )

        return event

    def log_data_access(self, actor: str, resource: str, purpose: str) -> AuditEvent:
        """Convenience method for data access events."""
        return self.log(
            event_type=EventType.DATA_ACCESS,
            actor=actor,
            resource=resource,
            action=f"accessed for {purpose}",
        )

    def log_breach(self, actor: str, details: dict) -> AuditEvent:
        """Log a potential security breach."""
        return self.log(
            event_type=EventType.BREACH_DETECTED,
            actor=actor,
            resource="system",
            action="potential breach detected",
            severity=Severity.CRITICAL,
            details=details,
        )

    def get_events(
        self,
        event_type: EventType | None = None,
        severity: Severity | None = None,
        actor: str | None = None,
    ) -> list[AuditEvent]:
        """Query events with optional filters."""
        results = self._events
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if severity:
            results = [e for e in results if e.severity == severity]
        if actor:
            results = [e for e in results if e.actor == actor]
        return results

    def generate_report(self) -> dict:
        """Generate a summary report of all audit events."""
        type_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        for event in self._events:
            t = event.event_type.value
            s = event.severity.value
            type_counts[t] = type_counts.get(t, 0) + 1
            severity_counts[s] = severity_counts.get(s, 0) + 1

        return {
            "total_events": len(self._events),
            "by_type": type_counts,
            "by_severity": severity_counts,
            "time_range": {
                "first": self._events[0].timestamp if self._events else None,
                "last": self._events[-1].timestamp if self._events else None,
            },
        }

    def verify_integrity(self) -> list[int]:
        """Check all events for tamper evidence.

        Returns:
            List of event indices that fail integrity verification.
        """
        failed: list[int] = []
        for i, event in enumerate(self._events):
            expected = event._compute_hash()
            if event.integrity_hash != expected:
                failed.append(i)
        return failed

    def _persist_event(self, event: AuditEvent) -> None:
        """Append event to the log file."""
        if self._log_path is None:
            return
        with open(self._log_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

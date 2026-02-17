"""Consent and Data Use Agreement (DUA) management.

Tracks patient consent status and DUA compliance for multi-site
federated learning studies, ensuring that data is only used in
accordance with approved protocols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class ConsentStatus(str, Enum):
    """Consent states for a patient-study pair."""

    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ConsentType(str, Enum):
    """Types of consent for clinical data use."""

    BROAD = "broad"
    STUDY_SPECIFIC = "study_specific"
    TIERED = "tiered"
    DYNAMIC = "dynamic"


@dataclass
class ConsentRecord:
    """Individual consent record.

    Attributes:
        patient_id: De-identified patient identifier.
        study_id: Clinical study identifier.
        consent_type: Category of consent granted.
        status: Current consent status.
        granted_at: Timestamp when consent was granted.
        expires_at: Optional expiration timestamp.
        revoked_at: Timestamp when consent was revoked (if applicable).
        data_uses: Permitted data use categories.
    """

    patient_id: str
    study_id: str
    consent_type: ConsentType = ConsentType.STUDY_SPECIFIC
    status: ConsentStatus = ConsentStatus.PENDING
    granted_at: str | None = None
    expires_at: str | None = None
    revoked_at: str | None = None
    data_uses: list[str] = field(default_factory=lambda: ["federated_learning", "model_training"])


class ConsentManager:
    """Manages patient consent and DUA compliance for federated studies.

    Provides methods to register, verify, and revoke consent,
    ensuring that the federated learning platform only processes
    data from consenting participants.
    """

    def __init__(self) -> None:
        self._records: dict[str, ConsentRecord] = {}
        self._audit_trail: list[dict] = []

    def register_consent(
        self,
        patient_id: str,
        study_id: str,
        consent_type: ConsentType | str = ConsentType.STUDY_SPECIFIC,
        data_uses: list[str] | None = None,
        expires_at: str | None = None,
    ) -> ConsentRecord:
        """Register patient consent for a study.

        Args:
            patient_id: De-identified patient identifier.
            study_id: Study identifier.
            consent_type: Type of consent being granted.
            data_uses: Permitted data use categories.
            expires_at: ISO-format expiration date (optional).

        Returns:
            The created ConsentRecord.
        """
        if isinstance(consent_type, str):
            consent_type = ConsentType(consent_type)

        key = f"{patient_id}:{study_id}"
        now = datetime.now(timezone.utc).isoformat()

        record = ConsentRecord(
            patient_id=patient_id,
            study_id=study_id,
            consent_type=consent_type,
            status=ConsentStatus.GRANTED,
            granted_at=now,
            expires_at=expires_at,
            data_uses=data_uses or ["federated_learning", "model_training"],
        )
        self._records[key] = record
        self._log_event("consent_granted", patient_id, study_id)

        logger.info(
            "Consent granted: patient=%s, study=%s, type=%s",
            patient_id,
            study_id,
            consent_type.value,
        )
        return record

    def verify_consent(
        self,
        patient_id: str,
        study_id: str,
        required_use: str = "federated_learning",
    ) -> bool:
        """Check whether a patient has active consent for a data use.

        Args:
            patient_id: De-identified patient identifier.
            study_id: Study identifier.
            required_use: The intended data use to verify.

        Returns:
            True if consent is active and covers the requested use.
        """
        key = f"{patient_id}:{study_id}"
        record = self._records.get(key)

        if record is None:
            return False

        if record.status != ConsentStatus.GRANTED:
            return False

        # Check expiration
        if record.expires_at:
            now = datetime.now(timezone.utc).isoformat()
            if now > record.expires_at:
                record.status = ConsentStatus.EXPIRED
                self._log_event("consent_expired", patient_id, study_id)
                return False

        if required_use not in record.data_uses:
            return False

        return True

    def revoke_consent(self, patient_id: str, study_id: str) -> bool:
        """Revoke a patient's consent for a study.

        Args:
            patient_id: De-identified patient identifier.
            study_id: Study identifier.

        Returns:
            True if consent was successfully revoked.
        """
        key = f"{patient_id}:{study_id}"
        record = self._records.get(key)

        if record is None:
            logger.warning("No consent found for %s in study %s", patient_id, study_id)
            return False

        record.status = ConsentStatus.REVOKED
        record.revoked_at = datetime.now(timezone.utc).isoformat()
        self._log_event("consent_revoked", patient_id, study_id)

        logger.info("Consent revoked: patient=%s, study=%s", patient_id, study_id)
        return True

    def get_consented_patients(self, study_id: str) -> list[str]:
        """List all patients with active consent for a study."""
        patients = []
        for key, record in self._records.items():
            if record.study_id == study_id and record.status == ConsentStatus.GRANTED:
                patients.append(record.patient_id)
        return patients

    def get_audit_trail(self) -> list[dict]:
        """Return the full consent audit trail."""
        return list(self._audit_trail)

    def _log_event(self, event: str, patient_id: str, study_id: str) -> None:
        """Record an event in the audit trail."""
        self._audit_trail.append(
            {
                "event": event,
                "patient_id": patient_id,
                "study_id": study_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

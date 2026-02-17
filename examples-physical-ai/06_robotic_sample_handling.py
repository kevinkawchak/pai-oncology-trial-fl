#!/usr/bin/env python3
"""Automated robotic sample handling with 21 CFR Part 11 audit trails.

CLINICAL CONTEXT
================
In oncology clinical trials, tissue specimens collected during surgical
procedures must follow a strict chain of custody from collection through
transport to pathology processing. 21 CFR Part 11 mandates electronic
records with full audit trails, electronic signatures, and access
controls to ensure specimen integrity and regulatory compliance.

This module demonstrates an automated robotic sample handling workflow
that integrates specimen collection, environmental monitoring during
transport, processing step tracking, and complete 21 CFR Part 11
compliant audit trails with electronic signature simulation.

USE CASES COVERED
=================
1. Specimen collection workflow with robotic pick-and-place, container
   assignment, and initial labelling.
2. Environmental monitoring during transport (temperature, humidity,
   vibration) with alert thresholds.
3. Chain-of-custody tracking with cryptographic integrity verification
   at each custody transfer.
4. 21 CFR Part 11 compliant audit trail with tamper-evident records,
   timestamps, and user attribution.
5. Electronic signature simulation with multi-factor authentication
   and signature meaning (e.g., 'verified', 'approved', 'released').
6. Processing workflow with step-by-step tracking, quality checks,
   and final pathology report generation.
7. Deviation and CAPA (corrective/preventive action) recording.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

Optional:
    torch >= 2.1.0   (https://pytorch.org) -- GPU-accelerated inference
    mujoco >= 3.0.0  (https://mujoco.org) -- Physics simulation backend

HARDWARE REQUIREMENTS
=====================
- CPU: 2+ cores
- RAM: 2 GB minimum
- GPU: Not required
- Robot: Sample handling manipulator (simulated)
- Sensors: Temperature, humidity, vibration (simulated)

REFERENCES
==========
- 21 CFR Part 11 -- Electronic Records; Electronic Signatures
- ISO 14971:2019 -- Risk management for process hazards
- IEC 62304:2006+AMD1:2015 -- Software traceability requirements
- CAP (College of American Pathologists) accreditation checklist

DISCLAIMER
==========
RESEARCH USE ONLY. This software is provided for research and educational
purposes. It is NOT validated for clinical use and must NOT be deployed in
any patient care setting without appropriate regulatory clearance and
comprehensive clinical validation.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]

HAS_MUJOCO = False
try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.sample_handling")


# ============================================================================
# Enumerations
# ============================================================================


class SpecimenType(str, Enum):
    """Types of tissue specimens in oncology procedures."""

    BIOPSY_CORE = "biopsy_core"
    RESECTION_MARGIN = "resection_margin"
    LYMPH_NODE = "lymph_node"
    FLUID_ASPIRATE = "fluid_aspirate"
    FROZEN_SECTION = "frozen_section"


class SpecimenStatus(str, Enum):
    """Status of a specimen in the handling pipeline."""

    COLLECTED = "collected"
    LABELLED = "labelled"
    IN_TRANSPORT = "in_transport"
    RECEIVED = "received"
    PROCESSING = "processing"
    ANALYSED = "analysed"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class CustodyAction(str, Enum):
    """Actions in the chain of custody."""

    COLLECTED = "collected"
    TRANSFERRED = "transferred"
    RECEIVED = "received"
    PROCESSED = "processed"
    RELEASED = "released"
    DISPOSED = "disposed"


class SignatureMeaning(str, Enum):
    """Meanings for electronic signatures per 21 CFR Part 11."""

    AUTHORED = "authored"
    VERIFIED = "verified"
    APPROVED = "approved"
    REVIEWED = "reviewed"
    RELEASED = "released"
    REJECTED = "rejected"


class AlertSeverity(str, Enum):
    """Severity of environmental or process alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ProcessingStep(str, Enum):
    """Steps in the specimen processing workflow."""

    ACCESSIONING = "accessioning"
    GROSSING = "grossing"
    FIXATION = "fixation"
    EMBEDDING = "embedding"
    SECTIONING = "sectioning"
    STAINING = "staining"
    COVERSLIPPING = "coverslipping"
    QC_REVIEW = "qc_review"
    PATHOLOGIST_REVIEW = "pathologist_review"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class AuditRecord:
    """A single 21 CFR Part 11 compliant audit trail record.

    Attributes:
        record_id: Unique record identifier (UUID).
        timestamp: ISO-format timestamp with timezone.
        unix_timestamp: Unix timestamp for ordering.
        user_id: Authenticated user who performed the action.
        action: Description of the action taken.
        resource_type: Type of resource affected.
        resource_id: Identifier of the affected resource.
        old_value: Previous value (for modifications).
        new_value: New value (for modifications).
        reason: Reason for the action.
        ip_address: Simulated IP address of the workstation.
        session_id: User session identifier.
        checksum: SHA-256 integrity hash.
        previous_checksum: Checksum of the previous record (chain).
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = ""
    unix_timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    action: str = ""
    resource_type: str = ""
    resource_id: str = ""
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    ip_address: str = "192.168.1.100"
    session_id: str = ""
    checksum: str = ""
    previous_checksum: str = ""

    def __post_init__(self) -> None:
        """Compute timestamp and integrity checksum."""
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(self.unix_timestamp))
        if not self.checksum:
            payload = (
                f"{self.record_id}:{self.unix_timestamp}:{self.user_id}:"
                f"{self.action}:{self.resource_id}:{self.previous_checksum}"
            )
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ElectronicSignatureRecord:
    """An electronic signature per 21 CFR Part 11.

    Attributes:
        signature_id: Unique signature identifier.
        timestamp: When the signature was applied.
        signer_id: User who signed.
        signer_name: Full name of the signer.
        meaning: Meaning of the signature.
        reason: Reason for signing.
        resource_id: What was signed.
        method: Authentication method used.
        checksum: Integrity hash.
    """

    signature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    signer_id: str = ""
    signer_name: str = ""
    meaning: SignatureMeaning = SignatureMeaning.AUTHORED
    reason: str = ""
    resource_id: str = ""
    method: str = "password+biometric"
    checksum: str = ""

    def __post_init__(self) -> None:
        """Compute integrity checksum."""
        if not self.checksum:
            payload = f"{self.signature_id}:{self.timestamp}:{self.signer_id}:{self.meaning.value}:{self.resource_id}"
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Specimen:
    """A tissue specimen in the handling pipeline.

    Attributes:
        specimen_id: Unique specimen identifier.
        specimen_type: Type of specimen.
        patient_id: De-identified patient identifier.
        trial_id: Clinical trial identifier.
        status: Current handling status.
        container_id: Container/cassette identifier.
        collection_time: When the specimen was collected.
        collection_site: Where the specimen was collected.
        weight_mg: Specimen weight in milligrams.
        dimensions_mm: Specimen dimensions (l, w, h).
        metadata: Additional specimen metadata.
    """

    specimen_id: str = field(default_factory=lambda: f"SPEC-{uuid.uuid4().hex[:8].upper()}")
    specimen_type: SpecimenType = SpecimenType.BIOPSY_CORE
    patient_id: str = ""
    trial_id: str = ""
    status: SpecimenStatus = SpecimenStatus.COLLECTED
    container_id: str = ""
    collection_time: float = field(default_factory=time.time)
    collection_site: str = ""
    weight_mg: float = 0.0
    dimensions_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentalReading:
    """Environmental sensor reading during specimen transport.

    Attributes:
        timestamp: Reading timestamp.
        temperature_c: Temperature in Celsius.
        humidity_percent: Relative humidity percentage.
        vibration_g: Vibration magnitude in g-force.
        location: Current location identifier.
        is_within_limits: Whether all readings are within limits.
    """

    timestamp: float = field(default_factory=time.time)
    temperature_c: float = 22.0
    humidity_percent: float = 45.0
    vibration_g: float = 0.01
    location: str = ""
    is_within_limits: bool = True


@dataclass
class CustodyEntry:
    """A chain-of-custody transfer record.

    Attributes:
        entry_id: Unique entry identifier.
        specimen_id: Specimen being tracked.
        action: Custody action taken.
        from_custodian: Person/system releasing custody.
        to_custodian: Person/system receiving custody.
        timestamp: Transfer timestamp.
        location: Where the transfer occurred.
        condition: Specimen condition at transfer.
        checksum: Integrity hash.
        previous_checksum: Previous entry checksum for chain integrity.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    specimen_id: str = ""
    action: CustodyAction = CustodyAction.COLLECTED
    from_custodian: str = ""
    to_custodian: str = ""
    timestamp: float = field(default_factory=time.time)
    location: str = ""
    condition: str = "intact"
    checksum: str = ""
    previous_checksum: str = ""

    def __post_init__(self) -> None:
        """Compute integrity checksum."""
        if not self.checksum:
            payload = (
                f"{self.entry_id}:{self.specimen_id}:{self.action.value}:"
                f"{self.from_custodian}:{self.to_custodian}:{self.previous_checksum}"
            )
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class DeviationRecord:
    """A recorded deviation from standard procedure.

    Attributes:
        deviation_id: Unique deviation identifier.
        timestamp: When the deviation was detected.
        specimen_id: Affected specimen.
        description: Description of the deviation.
        severity: Severity classification.
        corrective_action: Corrective action taken.
        preventive_action: Preventive action planned.
        resolved: Whether the deviation has been resolved.
    """

    deviation_id: str = field(default_factory=lambda: f"DEV-{uuid.uuid4().hex[:6].upper()}")
    timestamp: float = field(default_factory=time.time)
    specimen_id: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    corrective_action: str = ""
    preventive_action: str = ""
    resolved: bool = False


# ============================================================================
# Audit Trail Manager
# ============================================================================


class AuditTrailManager:
    """21 CFR Part 11 compliant audit trail management.

    Maintains a tamper-evident chain of audit records with
    cryptographic linking between consecutive entries.
    """

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._signatures: list[ElectronicSignatureRecord] = []
        self._session_counter = 0

    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        old_value: str = "",
        new_value: str = "",
        reason: str = "",
    ) -> AuditRecord:
        """Create a new audit record.

        Args:
            user_id: Authenticated user performing the action.
            action: Description of the action.
            resource_type: Type of resource affected.
            resource_id: Identifier of the affected resource.
            old_value: Previous value (for modifications).
            new_value: New value (for modifications).
            reason: Reason for the action.

        Returns:
            The created audit record.
        """
        previous_checksum = self._records[-1].checksum if self._records else ""

        record = AuditRecord(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            session_id=f"session_{self._session_counter}",
            previous_checksum=previous_checksum,
        )
        self._records.append(record)
        return record

    def add_signature(
        self,
        signer_id: str,
        signer_name: str,
        meaning: SignatureMeaning,
        resource_id: str,
        reason: str = "",
    ) -> ElectronicSignatureRecord:
        """Add an electronic signature.

        Args:
            signer_id: User ID of the signer.
            signer_name: Full name of the signer.
            meaning: Meaning of the signature.
            resource_id: What is being signed.
            reason: Reason for signing.

        Returns:
            The created signature record.
        """
        sig = ElectronicSignatureRecord(
            signer_id=signer_id,
            signer_name=signer_name,
            meaning=meaning,
            resource_id=resource_id,
            reason=reason,
        )
        self._signatures.append(sig)

        # Also log the signature in the audit trail
        self.log(
            user_id=signer_id,
            action=f"electronic_signature_{meaning.value}",
            resource_type="signature",
            resource_id=resource_id,
            reason=reason,
        )

        logger.info("E-Signature: %s (%s) -- %s for %s", signer_name, meaning.value, reason, resource_id)
        return sig

    def verify_chain_integrity(self) -> tuple[bool, list[int]]:
        """Verify the integrity of the audit trail chain.

        Returns:
            Tuple of (is_valid, list_of_broken_indices).
        """
        broken: list[int] = []

        for i, record in enumerate(self._records):
            # Verify checksum
            payload = (
                f"{record.record_id}:{record.unix_timestamp}:{record.user_id}:"
                f"{record.action}:{record.resource_id}:{record.previous_checksum}"
            )
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if record.checksum != expected:
                broken.append(i)
                continue

            # Verify chain linkage
            if i > 0 and record.previous_checksum != self._records[i - 1].checksum:
                broken.append(i)

        return len(broken) == 0, broken

    def get_records_for_resource(self, resource_id: str) -> list[AuditRecord]:
        """Get all audit records for a specific resource."""
        return [r for r in self._records if r.resource_id == resource_id]

    def get_signatures_for_resource(self, resource_id: str) -> list[ElectronicSignatureRecord]:
        """Get all signatures for a specific resource."""
        return [s for s in self._signatures if s.resource_id == resource_id]

    def generate_report(self) -> dict[str, Any]:
        """Generate an audit trail report."""
        is_valid, broken = self.verify_chain_integrity()
        return {
            "total_records": len(self._records),
            "total_signatures": len(self._signatures),
            "chain_integrity": is_valid,
            "broken_links": broken,
            "unique_users": len(set(r.user_id for r in self._records)),
            "actions_by_type": self._count_by_field("action"),
        }

    def _count_by_field(self, field_name: str) -> dict[str, int]:
        """Count records by a field value."""
        counts: dict[str, int] = {}
        for record in self._records:
            val = getattr(record, field_name, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts

    @property
    def record_count(self) -> int:
        """Total number of audit records."""
        return len(self._records)


# ============================================================================
# Chain of Custody
# ============================================================================


class ChainOfCustody:
    """Tracks specimen chain of custody with cryptographic integrity.

    Each custody transfer is linked to the previous entry via
    checksums, creating a tamper-evident chain.
    """

    def __init__(self, audit_trail: AuditTrailManager) -> None:
        self._entries: dict[str, list[CustodyEntry]] = {}
        self._audit = audit_trail

    def record_transfer(
        self,
        specimen_id: str,
        action: CustodyAction,
        from_custodian: str,
        to_custodian: str,
        location: str = "",
        condition: str = "intact",
    ) -> CustodyEntry:
        """Record a custody transfer.

        Args:
            specimen_id: Specimen being transferred.
            action: Type of custody action.
            from_custodian: Releasing party.
            to_custodian: Receiving party.
            location: Transfer location.
            condition: Specimen condition.

        Returns:
            The custody entry.
        """
        if specimen_id not in self._entries:
            self._entries[specimen_id] = []

        chain = self._entries[specimen_id]
        previous_checksum = chain[-1].checksum if chain else ""

        entry = CustodyEntry(
            specimen_id=specimen_id,
            action=action,
            from_custodian=from_custodian,
            to_custodian=to_custodian,
            location=location,
            condition=condition,
            previous_checksum=previous_checksum,
        )
        chain.append(entry)

        # Log in audit trail
        self._audit.log(
            user_id=to_custodian,
            action=f"custody_{action.value}",
            resource_type="specimen",
            resource_id=specimen_id,
            reason=f"Custody transfer at {location}",
        )

        logger.info(
            "Custody: %s -> %s (%s) specimen=%s at %s",
            from_custodian,
            to_custodian,
            action.value,
            specimen_id,
            location,
        )
        return entry

    def get_chain(self, specimen_id: str) -> list[CustodyEntry]:
        """Get the full custody chain for a specimen."""
        return list(self._entries.get(specimen_id, []))

    def verify_chain(self, specimen_id: str) -> tuple[bool, list[int]]:
        """Verify integrity of a specimen's custody chain.

        Returns:
            Tuple of (is_valid, broken_link_indices).
        """
        chain = self._entries.get(specimen_id, [])
        broken: list[int] = []

        for i, entry in enumerate(chain):
            # Recompute checksum
            payload = (
                f"{entry.entry_id}:{entry.specimen_id}:{entry.action.value}:"
                f"{entry.from_custodian}:{entry.to_custodian}:{entry.previous_checksum}"
            )
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if entry.checksum != expected:
                broken.append(i)
                continue

            if i > 0 and entry.previous_checksum != chain[i - 1].checksum:
                broken.append(i)

        return len(broken) == 0, broken

    def get_current_custodian(self, specimen_id: str) -> str:
        """Get the current custodian for a specimen."""
        chain = self._entries.get(specimen_id, [])
        if not chain:
            return "unknown"
        return chain[-1].to_custodian

    def get_summary(self) -> dict[str, Any]:
        """Get chain-of-custody summary."""
        return {
            "specimens_tracked": len(self._entries),
            "total_transfers": sum(len(chain) for chain in self._entries.values()),
            "chain_integrity": {sid: self.verify_chain(sid)[0] for sid in self._entries},
        }


# ============================================================================
# Environmental Monitor
# ============================================================================


class EnvironmentalMonitor:
    """Monitors environmental conditions during specimen transport.

    Tracks temperature, humidity, and vibration with configurable
    alert thresholds.
    """

    def __init__(
        self,
        temp_range_c: tuple[float, float] = (2.0, 8.0),
        humidity_range_pct: tuple[float, float] = (30.0, 60.0),
        max_vibration_g: float = 0.5,
    ) -> None:
        self._temp_range = temp_range_c
        self._humidity_range = humidity_range_pct
        self._max_vibration = max_vibration_g
        self._readings: list[EnvironmentalReading] = []
        self._alerts: list[dict[str, Any]] = []

    def record_reading(self, reading: EnvironmentalReading) -> list[dict[str, Any]]:
        """Record an environmental reading and check limits.

        Args:
            reading: Environmental sensor reading.

        Returns:
            List of alerts triggered by this reading.
        """
        alerts: list[dict[str, Any]] = []

        # Temperature check
        if reading.temperature_c < self._temp_range[0] or reading.temperature_c > self._temp_range[1]:
            alert = {
                "type": "temperature",
                "severity": AlertSeverity.CRITICAL.value,
                "value": reading.temperature_c,
                "range": self._temp_range,
                "message": f"Temperature {reading.temperature_c:.1f}C outside range {self._temp_range}",
            }
            alerts.append(alert)
            reading.is_within_limits = False

        # Humidity check
        if reading.humidity_percent < self._humidity_range[0] or reading.humidity_percent > self._humidity_range[1]:
            alert = {
                "type": "humidity",
                "severity": AlertSeverity.WARNING.value,
                "value": reading.humidity_percent,
                "range": self._humidity_range,
                "message": f"Humidity {reading.humidity_percent:.1f}% outside range {self._humidity_range}",
            }
            alerts.append(alert)
            reading.is_within_limits = False

        # Vibration check
        if reading.vibration_g > self._max_vibration:
            alert = {
                "type": "vibration",
                "severity": AlertSeverity.WARNING.value,
                "value": reading.vibration_g,
                "limit": self._max_vibration,
                "message": f"Vibration {reading.vibration_g:.3f}g exceeds limit {self._max_vibration}g",
            }
            alerts.append(alert)
            reading.is_within_limits = False

        self._readings.append(reading)
        self._alerts.extend(alerts)

        for alert in alerts:
            logger.warning("ENV ALERT: %s", alert["message"])

        return alerts

    def get_summary(self) -> dict[str, Any]:
        """Get environmental monitoring summary."""
        if not self._readings:
            return {"reading_count": 0}

        temps = [r.temperature_c for r in self._readings]
        humidities = [r.humidity_percent for r in self._readings]
        vibrations = [r.vibration_g for r in self._readings]
        within_limits = sum(1 for r in self._readings if r.is_within_limits)

        return {
            "reading_count": len(self._readings),
            "within_limits_pct": within_limits / len(self._readings) * 100,
            "temperature": {
                "mean": float(np.mean(temps)),
                "min": float(np.min(temps)),
                "max": float(np.max(temps)),
            },
            "humidity": {
                "mean": float(np.mean(humidities)),
                "min": float(np.min(humidities)),
                "max": float(np.max(humidities)),
            },
            "vibration": {
                "mean": float(np.mean(vibrations)),
                "max": float(np.max(vibrations)),
            },
            "total_alerts": len(self._alerts),
        }


# ============================================================================
# Sample Handling Workflow
# ============================================================================


class SampleHandlingWorkflow:
    """Complete robotic sample handling workflow orchestrator.

    Manages the end-to-end specimen handling process from collection
    through processing with full audit trail and chain of custody.
    """

    def __init__(self, trial_id: str = "ONCO-TRIAL-001") -> None:
        self._trial_id = trial_id
        self._audit = AuditTrailManager()
        self._custody = ChainOfCustody(self._audit)
        self._env_monitor = EnvironmentalMonitor()
        self._specimens: dict[str, Specimen] = {}
        self._deviations: list[DeviationRecord] = []
        self._processing_log: dict[str, list[dict[str, Any]]] = {}

        self._audit.log(
            user_id="system",
            action="workflow_initialized",
            resource_type="workflow",
            resource_id=trial_id,
            reason="Sample handling workflow initialized",
        )
        logger.info("SampleHandlingWorkflow initialized for trial '%s'", trial_id)

    def collect_specimen(
        self,
        specimen_type: SpecimenType,
        patient_id: str,
        collector_id: str,
        collection_site: str = "OR-1",
        weight_mg: float = 50.0,
        dimensions_mm: tuple[float, float, float] = (10.0, 5.0, 3.0),
    ) -> Specimen:
        """Collect a new specimen.

        Args:
            specimen_type: Type of specimen.
            patient_id: De-identified patient ID.
            collector_id: ID of the collecting clinician.
            collection_site: Where collection occurred.
            weight_mg: Specimen weight.
            dimensions_mm: Specimen dimensions.

        Returns:
            The created specimen.
        """
        specimen = Specimen(
            specimen_type=specimen_type,
            patient_id=patient_id,
            trial_id=self._trial_id,
            status=SpecimenStatus.COLLECTED,
            container_id=f"CONT-{uuid.uuid4().hex[:6].upper()}",
            collection_site=collection_site,
            weight_mg=weight_mg,
            dimensions_mm=dimensions_mm,
        )
        self._specimens[specimen.specimen_id] = specimen
        self._processing_log[specimen.specimen_id] = []

        # Audit trail
        self._audit.log(
            user_id=collector_id,
            action="specimen_collected",
            resource_type="specimen",
            resource_id=specimen.specimen_id,
            new_value=f"type={specimen_type.value}, weight={weight_mg}mg",
            reason=f"Specimen collection during procedure at {collection_site}",
        )

        # Chain of custody
        self._custody.record_transfer(
            specimen_id=specimen.specimen_id,
            action=CustodyAction.COLLECTED,
            from_custodian="patient",
            to_custodian=collector_id,
            location=collection_site,
        )

        # Electronic signature for collection
        self._audit.add_signature(
            signer_id=collector_id,
            signer_name=f"Dr. {collector_id.replace('_', ' ').title()}",
            meaning=SignatureMeaning.AUTHORED,
            resource_id=specimen.specimen_id,
            reason="Specimen collection verification",
        )

        logger.info(
            "Specimen collected: %s (%s) from patient %s",
            specimen.specimen_id,
            specimen_type.value,
            patient_id,
        )
        return specimen

    def label_specimen(self, specimen_id: str, labeller_id: str) -> bool:
        """Apply label to a specimen container.

        Args:
            specimen_id: Specimen to label.
            labeller_id: Person applying the label.

        Returns:
            True if labelling succeeded.
        """
        specimen = self._specimens.get(specimen_id)
        if specimen is None:
            logger.error("Specimen %s not found", specimen_id)
            return False

        old_status = specimen.status.value
        specimen.status = SpecimenStatus.LABELLED

        self._audit.log(
            user_id=labeller_id,
            action="specimen_labelled",
            resource_type="specimen",
            resource_id=specimen_id,
            old_value=old_status,
            new_value=SpecimenStatus.LABELLED.value,
        )

        logger.info("Specimen %s labelled by %s", specimen_id, labeller_id)
        return True

    def transport_specimen(
        self,
        specimen_id: str,
        transporter_id: str,
        from_location: str,
        to_location: str,
        num_readings: int = 10,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Transport a specimen with environmental monitoring.

        Args:
            specimen_id: Specimen to transport.
            transporter_id: Person/robot performing transport.
            from_location: Starting location.
            to_location: Destination location.
            num_readings: Number of environmental readings to simulate.
            seed: Random seed.

        Returns:
            List of environmental alerts during transport.
        """
        specimen = self._specimens.get(specimen_id)
        if specimen is None:
            logger.error("Specimen %s not found", specimen_id)
            return []

        specimen.status = SpecimenStatus.IN_TRANSPORT

        # Custody transfer to transporter
        self._custody.record_transfer(
            specimen_id=specimen_id,
            action=CustodyAction.TRANSFERRED,
            from_custodian=self._custody.get_current_custodian(specimen_id),
            to_custodian=transporter_id,
            location=from_location,
        )

        # Simulate environmental readings during transport
        rng = np.random.default_rng(seed)
        all_alerts: list[dict[str, Any]] = []
        base_time = time.time()

        for i in range(num_readings):
            temp = 4.0 + rng.normal(0, 1.0)
            humidity = 45.0 + rng.normal(0, 5.0)
            vibration = float(rng.exponential(0.05))

            # Inject one anomaly
            if i == num_readings // 2:
                temp = 10.0  # Temperature excursion

            reading = EnvironmentalReading(
                timestamp=base_time + i * 30,
                temperature_c=temp,
                humidity_percent=humidity,
                vibration_g=vibration,
                location=f"transit_{from_location}_to_{to_location}",
            )
            alerts = self._env_monitor.record_reading(reading)
            all_alerts.extend(alerts)

        # Record deviations for environmental alerts
        if all_alerts:
            deviation = DeviationRecord(
                specimen_id=specimen_id,
                description=f"{len(all_alerts)} environmental alerts during transport",
                severity=AlertSeverity.WARNING,
                corrective_action="Specimen inspected upon receipt; transport conditions documented",
                preventive_action="Recalibrate transport container temperature control",
            )
            self._deviations.append(deviation)

            self._audit.log(
                user_id=transporter_id,
                action="deviation_recorded",
                resource_type="deviation",
                resource_id=deviation.deviation_id,
                reason=deviation.description,
            )

        # Arrival
        self._custody.record_transfer(
            specimen_id=specimen_id,
            action=CustodyAction.RECEIVED,
            from_custodian=transporter_id,
            to_custodian="pathology_lab",
            location=to_location,
        )

        specimen.status = SpecimenStatus.RECEIVED

        self._audit.log(
            user_id="pathology_lab",
            action="specimen_received",
            resource_type="specimen",
            resource_id=specimen_id,
            new_value=f"received at {to_location}",
        )

        logger.info(
            "Specimen %s transported from %s to %s (%d alerts)",
            specimen_id,
            from_location,
            to_location,
            len(all_alerts),
        )
        return all_alerts

    def process_specimen(self, specimen_id: str, technician_id: str, seed: int = 42) -> dict[str, Any]:
        """Process a specimen through all pathology steps.

        Args:
            specimen_id: Specimen to process.
            technician_id: Processing technician ID.
            seed: Random seed for simulation.

        Returns:
            Processing results.
        """
        specimen = self._specimens.get(specimen_id)
        if specimen is None:
            return {"success": False, "error": "Specimen not found"}

        rng = np.random.default_rng(seed)
        specimen.status = SpecimenStatus.PROCESSING

        self._audit.log(
            user_id=technician_id,
            action="processing_started",
            resource_type="specimen",
            resource_id=specimen_id,
        )

        processing_results: list[dict[str, Any]] = []
        steps = list(ProcessingStep)

        for step in steps:
            step_start = time.time()
            duration_s = float(rng.exponential(120))
            quality_score = float(rng.beta(8, 2))  # Mostly high quality
            passed = quality_score > 0.3

            step_result = {
                "step": step.value,
                "duration_s": round(duration_s, 2),
                "quality_score": round(quality_score, 4),
                "passed": passed,
                "technician": technician_id,
                "timestamp": step_start,
            }
            processing_results.append(step_result)
            self._processing_log[specimen_id].append(step_result)

            self._audit.log(
                user_id=technician_id,
                action=f"processing_step_{step.value}",
                resource_type="specimen",
                resource_id=specimen_id,
                new_value=f"quality={quality_score:.4f}, passed={passed}",
            )

            if not passed:
                deviation = DeviationRecord(
                    specimen_id=specimen_id,
                    description=f"Quality check failed at step '{step.value}' (score={quality_score:.4f})",
                    severity=AlertSeverity.CRITICAL,
                    corrective_action=f"Repeat {step.value} step",
                )
                self._deviations.append(deviation)

            logger.info("  Step [%s]: quality=%.4f %s", step.value, quality_score, "PASS" if passed else "FAIL")

        # Pathologist review and signature
        specimen.status = SpecimenStatus.ANALYSED

        self._audit.add_signature(
            signer_id=technician_id,
            signer_name=f"Tech. {technician_id.replace('_', ' ').title()}",
            meaning=SignatureMeaning.VERIFIED,
            resource_id=specimen_id,
            reason="Processing verification -- all steps completed",
        )

        pathologist_id = "dr_pathologist_01"
        self._audit.add_signature(
            signer_id=pathologist_id,
            signer_name="Dr. Sarah Chen, MD",
            meaning=SignatureMeaning.REVIEWED,
            resource_id=specimen_id,
            reason="Pathology review completed",
        )

        self._audit.add_signature(
            signer_id=pathologist_id,
            signer_name="Dr. Sarah Chen, MD",
            meaning=SignatureMeaning.RELEASED,
            resource_id=specimen_id,
            reason="Results released for clinical use",
        )

        # Custody transfer for processing completion
        self._custody.record_transfer(
            specimen_id=specimen_id,
            action=CustodyAction.PROCESSED,
            from_custodian="pathology_lab",
            to_custodian="archive",
            location="pathology_lab",
        )

        all_passed = all(r["passed"] for r in processing_results)
        return {
            "success": True,
            "specimen_id": specimen_id,
            "steps_completed": len(processing_results),
            "all_passed": all_passed,
            "processing_results": processing_results,
        }

    def archive_specimen(self, specimen_id: str, archivist_id: str) -> bool:
        """Archive a specimen after processing.

        Args:
            specimen_id: Specimen to archive.
            archivist_id: Person performing archival.

        Returns:
            True if archival succeeded.
        """
        specimen = self._specimens.get(specimen_id)
        if specimen is None:
            return False

        specimen.status = SpecimenStatus.ARCHIVED

        self._custody.record_transfer(
            specimen_id=specimen_id,
            action=CustodyAction.RELEASED,
            from_custodian="archive",
            to_custodian="biobank_storage",
            location="biobank",
            condition="preserved",
        )

        self._audit.log(
            user_id=archivist_id,
            action="specimen_archived",
            resource_type="specimen",
            resource_id=specimen_id,
            reason="Specimen archived in biobank for long-term storage",
        )

        self._audit.add_signature(
            signer_id=archivist_id,
            signer_name=f"Tech. {archivist_id.replace('_', ' ').title()}",
            meaning=SignatureMeaning.APPROVED,
            resource_id=specimen_id,
            reason="Archival approval",
        )

        logger.info("Specimen %s archived", specimen_id)
        return True

    def get_report(self) -> dict[str, Any]:
        """Generate a comprehensive sample handling report."""
        audit_report = self._audit.generate_report()
        custody_summary = self._custody.get_summary()
        env_summary = self._env_monitor.get_summary()

        return {
            "trial_id": self._trial_id,
            "total_specimens": len(self._specimens),
            "specimen_status": {sid: spec.status.value for sid, spec in self._specimens.items()},
            "audit_trail": audit_report,
            "chain_of_custody": custody_summary,
            "environmental_monitoring": env_summary,
            "deviations": len(self._deviations),
            "deviation_details": [
                {
                    "id": d.deviation_id,
                    "specimen": d.specimen_id,
                    "description": d.description,
                    "severity": d.severity.value,
                }
                for d in self._deviations
            ],
        }


# ============================================================================
# Demonstration
# ============================================================================


def run_sample_handling_demo() -> dict[str, Any]:
    """Run the complete robotic sample handling demonstration.

    Returns:
        Final workflow report.
    """
    logger.info("=" * 70)
    logger.info("  Robotic Sample Handling Demo (21 CFR Part 11)")
    logger.info("  Version 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 70)

    workflow = SampleHandlingWorkflow(trial_id="ONCO-FL-2026-001")

    # 1. Collect specimens
    logger.info("\n--- Phase 1: Specimen Collection ---")
    specimens_to_collect = [
        (SpecimenType.BIOPSY_CORE, "PAT-001", "dr_surgeon_01", "OR-1", 45.0),
        (SpecimenType.RESECTION_MARGIN, "PAT-001", "dr_surgeon_01", "OR-1", 120.0),
        (SpecimenType.LYMPH_NODE, "PAT-001", "dr_surgeon_01", "OR-1", 80.0),
        (SpecimenType.FROZEN_SECTION, "PAT-002", "dr_surgeon_02", "OR-3", 30.0),
    ]

    collected_specimens: list[Specimen] = []
    for spec_type, pat_id, collector, site, weight in specimens_to_collect:
        specimen = workflow.collect_specimen(
            specimen_type=spec_type,
            patient_id=pat_id,
            collector_id=collector,
            collection_site=site,
            weight_mg=weight,
        )
        collected_specimens.append(specimen)

    logger.info("Collected %d specimens", len(collected_specimens))

    # 2. Label specimens
    logger.info("\n--- Phase 2: Specimen Labelling ---")
    for specimen in collected_specimens:
        workflow.label_specimen(specimen.specimen_id, "tech_labeller_01")

    # 3. Transport specimens
    logger.info("\n--- Phase 3: Specimen Transport ---")
    total_alerts = 0
    for i, specimen in enumerate(collected_specimens):
        alerts = workflow.transport_specimen(
            specimen_id=specimen.specimen_id,
            transporter_id="robot_transport_01",
            from_location=specimen.collection_site,
            to_location="pathology_lab",
            num_readings=10,
            seed=42 + i,
        )
        total_alerts += len(alerts)

    logger.info("Transport complete. Total environmental alerts: %d", total_alerts)

    # 4. Process specimens
    logger.info("\n--- Phase 4: Specimen Processing ---")
    for i, specimen in enumerate(collected_specimens):
        logger.info("\nProcessing specimen %s (%s):", specimen.specimen_id, specimen.specimen_type.value)
        result = workflow.process_specimen(
            specimen_id=specimen.specimen_id,
            technician_id="tech_processor_01",
            seed=42 + i * 10,
        )
        logger.info(
            "  Result: %s (%d steps, all_passed=%s)",
            "SUCCESS" if result["success"] else "FAILED",
            result.get("steps_completed", 0),
            result.get("all_passed", False),
        )

    # 5. Archive specimens
    logger.info("\n--- Phase 5: Specimen Archival ---")
    for specimen in collected_specimens:
        workflow.archive_specimen(specimen.specimen_id, "tech_archivist_01")

    # 6. Generate final report
    logger.info("\n--- Final Report ---")
    report = workflow.get_report()

    logger.info("  Trial ID:              %s", report["trial_id"])
    logger.info("  Total specimens:       %d", report["total_specimens"])
    logger.info("  Specimen statuses:")
    for sid, status in report["specimen_status"].items():
        logger.info("    %-20s %s", sid, status)

    audit = report["audit_trail"]
    logger.info("\n  Audit Trail:")
    logger.info("    Total records:       %d", audit["total_records"])
    logger.info("    Total signatures:    %d", audit["total_signatures"])
    logger.info("    Chain integrity:     %s", audit["chain_integrity"])
    logger.info("    Unique users:        %d", audit["unique_users"])

    custody = report["chain_of_custody"]
    logger.info("\n  Chain of Custody:")
    logger.info("    Specimens tracked:   %d", custody["specimens_tracked"])
    logger.info("    Total transfers:     %d", custody["total_transfers"])
    for sid, is_valid in custody["chain_integrity"].items():
        logger.info("    %-20s %s", sid, "VALID" if is_valid else "BROKEN")

    env = report["environmental_monitoring"]
    if env.get("reading_count", 0) > 0:
        logger.info("\n  Environmental Monitoring:")
        logger.info("    Readings:            %d", env["reading_count"])
        logger.info("    Within limits:       %.1f%%", env["within_limits_pct"])
        logger.info("    Temp range:          %.1f - %.1fC", env["temperature"]["min"], env["temperature"]["max"])
        logger.info("    Alerts:              %d", env["total_alerts"])

    logger.info("\n  Deviations:            %d", report["deviations"])
    for dev in report["deviation_details"]:
        logger.info("    [%s] %s: %s", dev["severity"], dev["id"], dev["description"])

    logger.info("\n" + "=" * 70)
    logger.info("  RESEARCH USE ONLY -- NOT FOR CLINICAL DEPLOYMENT")
    logger.info("=" * 70)

    return report


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    report = run_sample_handling_demo()
    sys.exit(0)

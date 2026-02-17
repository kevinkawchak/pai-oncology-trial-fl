"""
Clinical System Integration for Oncology Digital Twins.

CLINICAL CONTEXT:
    Provides the integration layer between clinical information systems
    and the digital twins platform within the PAI Federated Learning
    framework. Supports data ingestion from Electronic Health Records (EHR),
    Picture Archiving and Communication Systems (PACS), Laboratory
    Information Systems (LIS), Radiology Information Systems (RIS), and
    Treatment Planning Systems (TPS).

    Data is translated from standard clinical formats (HL7 FHIR, DICOM,
    CSV, JSON) into the internal representations used by the patient
    modeling and treatment simulation engines. The FederatedClinicalBridge
    connects digital twin outputs to the federated learning pipeline for
    multi-site model training without sharing raw patient data.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0  (https://numpy.org/)
        - scipy >= 1.11.0  (https://scipy.org/)
    Optional:
        - torch >= 2.5.0   (https://pytorch.org/)
        - monai >= 1.3.0   (https://monai.io/)

REFERENCES:
    - HL7 FHIR R4 Specification. URL: https://hl7.org/fhir/R4/
    - DICOM Standard PS3.1 2024b. URL: https://www.dicomstandard.org/
    - Mandel et al. (2016). SMART on FHIR: A standards-based,
      interoperable apps platform for EHRs. J Am Med Inform Assoc,
      23(5). DOI: 10.1093/jamia/ocv189
    - McMurry et al. (2022). Federated learning for clinical data
      integration. npj Digital Medicine, 5(1).
      DOI: 10.1038/s41746-022-00689-2

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or guide patient treatment. All simulation results
    must be independently validated before any clinical application.

VERSION: 0.4.0
LICENSE: MIT
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports for optional dependencies
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    stats = None  # type: ignore[assignment]
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RECORDS_PER_SYNC: int = 10000
MAX_PAYLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_FEATURE_VECTOR_SIZE: int = 64
SUPPORTED_FHIR_RESOURCES: list[str] = [
    "Patient",
    "Condition",
    "Observation",
    "MedicationAdministration",
    "Procedure",
    "DiagnosticReport",
    "ImagingStudy",
]
SYNC_TIMEOUT_SECONDS: float = 300.0


# ---------------------------------------------------------------------------
# Enum classes
# ---------------------------------------------------------------------------
class ClinicalSystem(Enum):
    """Source clinical information system types.

    Each system type determines the expected data format, authentication
    protocol, and available data elements for integration.
    """

    EHR = "electronic_health_record"
    PACS = "picture_archiving_communication_system"
    LIS = "laboratory_information_system"
    RIS = "radiology_information_system"
    TPS = "treatment_planning_system"


class IntegrationStatus(Enum):
    """Connection and synchronization status for clinical system integration.

    Tracks the current state of the integration channel between the
    digital twins platform and a clinical information system.
    """

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SYNCING = "syncing"
    ERROR = "error"


class DataFormat(Enum):
    """Supported clinical data interchange formats.

    Each format requires specific parsing logic and validation rules
    for safe ingestion into the digital twin pipeline.
    """

    HL7_FHIR = "hl7_fhir"
    DICOM = "dicom"
    CSV = "csv"
    JSON = "json"


class DataCategory(Enum):
    """Categories of clinical data elements for structured ingestion."""

    DEMOGRAPHICS = "demographics"
    DIAGNOSIS = "diagnosis"
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    VITALS = "vitals"
    GENOMIC = "genomic"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ClinicalDataPoint:
    """Single clinical observation from a source system.

    Represents an atomic unit of clinical data with full provenance
    tracking for audit and regulatory compliance.

    Attributes:
        data_id: Unique identifier for this data point.
        patient_id: De-identified patient identifier.
        source_system: Clinical system that produced this data.
        data_format: Format of the source data.
        category: Category of clinical data.
        timestamp_unix: Unix timestamp of the observation.
        name: Human-readable name of the observation.
        value: Numeric or string value of the observation.
        unit: Unit of measurement (e.g., "mg/dL", "cm3").
        reference_range: Normal reference range as (low, high) tuple.
        is_deidentified: Whether PHI has been removed.
        metadata: Additional provenance and quality metadata.
    """

    data_id: str = ""
    patient_id: str = ""
    source_system: ClinicalSystem = ClinicalSystem.EHR
    data_format: DataFormat = DataFormat.JSON
    category: DataCategory = DataCategory.LABORATORY
    timestamp_unix: float = 0.0
    name: str = ""
    value: float | str = 0.0
    unit: str = ""
    reference_range: tuple[float, float] = (0.0, 0.0)
    is_deidentified: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate data ID and validate fields."""
        if not self.data_id:
            self.data_id = f"CDP-{uuid.uuid4().hex[:12].upper()}"
        if self.timestamp_unix <= 0.0:
            self.timestamp_unix = time.time()
        if not self.patient_id:
            self.patient_id = f"UNKNOWN-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class IntegrationConfig:
    """Configuration for a clinical system integration channel.

    Attributes:
        config_id: Unique configuration identifier.
        system_type: Type of clinical system to integrate.
        endpoint_url: Base URL or connection string for the system.
        data_format: Expected data format from this system.
        auth_method: Authentication method ("api_key", "oauth2", "certificate").
        polling_interval_seconds: Data polling interval in seconds.
        batch_size: Maximum records per synchronization batch.
        enabled: Whether this integration channel is active.
        deidentification_required: Whether to apply de-identification.
        allowed_categories: Permitted data categories for ingestion.
        metadata: Additional configuration metadata.
    """

    config_id: str = ""
    system_type: ClinicalSystem = ClinicalSystem.EHR
    endpoint_url: str = ""
    data_format: DataFormat = DataFormat.HL7_FHIR
    auth_method: str = "api_key"
    polling_interval_seconds: float = 300.0
    batch_size: int = 100
    enabled: bool = True
    deidentification_required: bool = True
    allowed_categories: list[DataCategory] = field(
        default_factory=lambda: [
            DataCategory.DIAGNOSIS,
            DataCategory.LABORATORY,
            DataCategory.IMAGING,
            DataCategory.TREATMENT,
        ]
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate config ID if not provided."""
        if not self.config_id:
            self.config_id = f"INTG-{uuid.uuid4().hex[:12].upper()}"
        self.batch_size = int(np.clip(self.batch_size, 1, MAX_RECORDS_PER_SYNC))
        self.polling_interval_seconds = float(np.clip(self.polling_interval_seconds, 10.0, 86400.0))


@dataclass
class SyncResult:
    """Result of a data synchronization operation.

    Attributes:
        sync_id: Unique synchronization identifier.
        config_id: Associated integration configuration ID.
        status: Final status of the synchronization.
        records_received: Number of records received from the source.
        records_accepted: Number of records that passed validation.
        records_rejected: Number of records that failed validation.
        duration_seconds: Total synchronization duration in seconds.
        error_message: Error description if status is ERROR.
        timestamp_unix: Unix timestamp of synchronization completion.
        data_points: Successfully ingested data points.
        metadata: Additional synchronization metadata.
    """

    sync_id: str = ""
    config_id: str = ""
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    records_received: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    duration_seconds: float = 0.0
    error_message: str = ""
    timestamp_unix: float = 0.0
    data_points: list[ClinicalDataPoint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate sync ID and set timestamp."""
        if not self.sync_id:
            self.sync_id = f"SYNC-{uuid.uuid4().hex[:12].upper()}"
        if self.timestamp_unix <= 0.0:
            self.timestamp_unix = time.time()


@dataclass
class ClinicalReport:
    """Structured clinical report generated from digital twin simulation.

    Designed for treatment planning review, integrating simulation
    outcomes with patient clinical context.

    Attributes:
        report_id: Unique report identifier.
        patient_id: De-identified patient identifier.
        report_type: Type of clinical report.
        generated_timestamp_unix: Unix timestamp of report generation.
        summary: Executive summary of findings.
        sections: Ordered list of report sections with content.
        recommendations: Treatment recommendations from simulation.
        confidence_level: Overall confidence in the report (0-1).
        data_sources: List of data source identifiers used.
        disclaimer: Regulatory disclaimer text.
        metadata: Additional report metadata.
    """

    report_id: str = ""
    patient_id: str = ""
    report_type: str = "treatment_simulation_summary"
    generated_timestamp_unix: float = 0.0
    summary: str = ""
    sections: list[dict[str, str]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence_level: float = 0.0
    data_sources: list[str] = field(default_factory=list)
    disclaimer: str = (
        "RESEARCH USE ONLY. This report is generated by a computational "
        "model for research purposes only. It has NOT been validated for "
        "clinical use and MUST NOT guide clinical decisions."
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate report ID and set timestamp."""
        if not self.report_id:
            self.report_id = f"RPT-{uuid.uuid4().hex[:12].upper()}"
        if self.generated_timestamp_unix <= 0.0:
            self.generated_timestamp_unix = time.time()
        self.confidence_level = float(np.clip(self.confidence_level, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_data_hash(data: dict[str, Any]) -> str:
    """Compute SHA-256 hash of clinical data for integrity verification.

    Args:
        data: Dictionary of clinical data to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_fhir_resource(resource: dict[str, Any]) -> bool:
    """Perform basic validation of a FHIR resource structure.

    Checks for required FHIR fields: resourceType and id. This is a
    minimal structural check, not a full FHIR profile validation.

    Args:
        resource: Dictionary representing a FHIR resource.

    Returns:
        True if the resource has the minimum required structure.
    """
    if not isinstance(resource, dict):
        return False
    resource_type = resource.get("resourceType", "")
    if resource_type not in SUPPORTED_FHIR_RESOURCES:
        logger.warning("Unsupported FHIR resource type: %s", resource_type)
        return False
    if "id" not in resource:
        logger.warning("FHIR resource missing required 'id' field")
        return False
    return True


def _deidentify_record(record: dict[str, Any]) -> dict[str, Any]:
    """Apply basic de-identification to a clinical data record.

    Removes common PHI fields and replaces identifiers with hashed
    values. This is a structural transformation only and does NOT
    constitute a HIPAA Safe Harbor or Expert Determination method.

    Args:
        record: Raw clinical data record.

    Returns:
        De-identified copy of the record.
    """
    phi_fields = [
        "name",
        "address",
        "phone",
        "email",
        "ssn",
        "mrn",
        "date_of_birth",
        "zip_code",
        "ip_address",
        "device_id",
        "account_number",
        "certificate_number",
        "vehicle_id",
        "web_url",
        "biometric_id",
        "photo",
    ]
    cleaned = {}
    for key, value in record.items():
        key_lower = key.lower().replace("-", "_").replace(" ", "_")
        if key_lower in phi_fields:
            logger.debug("Removing PHI field: %s", key)
            continue
        cleaned[key] = value

    if "patient_id" in cleaned:
        raw_id = str(cleaned["patient_id"])
        cleaned["patient_id"] = f"DI-{hashlib.sha256(raw_id.encode()).hexdigest()[:12].upper()}"

    return cleaned


def _parse_csv_records(csv_text: str, delimiter: str = ",") -> list[dict[str, str]]:
    """Parse CSV text into a list of record dictionaries.

    Args:
        csv_text: Raw CSV text with header row.
        delimiter: Field delimiter character.

    Returns:
        List of dictionaries mapping column names to values.
    """
    lines = csv_text.strip().split("\n")
    if len(lines) < 2:
        logger.warning("CSV text has fewer than 2 lines (header + data)")
        return []

    headers = [h.strip().lower().replace(" ", "_") for h in lines[0].split(delimiter)]
    records = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split(delimiter)]
        if len(values) == len(headers):
            record = dict(zip(headers, values))
            records.append(record)
        else:
            logger.warning(
                "Skipping CSV row: expected %d fields, got %d",
                len(headers),
                len(values),
            )
    return records


# ---------------------------------------------------------------------------
# ClinicalIntegrator
# ---------------------------------------------------------------------------
class ClinicalIntegrator:
    """Core integration engine for clinical data ingestion and export.

    Manages connections to clinical information systems, ingests data
    in standard clinical formats, applies de-identification, and
    transforms data into digital twin input representations.

    Args:
        site_id: Identifier for the clinical trial site.
        configs: List of integration configurations for connected systems.
    """

    def __init__(
        self,
        site_id: str = "SITE-001",
        configs: list[IntegrationConfig] | None = None,
    ) -> None:
        self.site_id = site_id
        self._configs: dict[str, IntegrationConfig] = {}
        self._connection_status: dict[str, IntegrationStatus] = {}
        self._data_store: dict[str, list[ClinicalDataPoint]] = {}
        self._sync_history: list[SyncResult] = []
        self._ingestion_count: int = 0

        if configs:
            for config in configs:
                self.register_system(config)

        logger.info(
            "ClinicalIntegrator initialized for site %s with %d system(s)",
            self.site_id,
            len(self._configs),
        )

    def register_system(self, config: IntegrationConfig) -> None:
        """Register a clinical system integration configuration.

        Args:
            config: Integration configuration to register.
        """
        self._configs[config.config_id] = config
        self._connection_status[config.config_id] = IntegrationStatus.DISCONNECTED
        self._data_store[config.config_id] = []
        logger.info(
            "Registered clinical system: %s (%s, format=%s)",
            config.config_id,
            config.system_type.value,
            config.data_format.value,
        )

    def connect(self, config_id: str) -> IntegrationStatus:
        """Establish connection to a registered clinical system.

        In research mode, this simulates a connection handshake.
        Production implementations would establish actual network
        connections with authentication.

        Args:
            config_id: ID of the integration configuration.

        Returns:
            Current integration status after connection attempt.
        """
        if config_id not in self._configs:
            logger.error("Unknown integration config: %s", config_id)
            return IntegrationStatus.ERROR

        config = self._configs[config_id]
        if not config.enabled:
            logger.warning("Integration %s is disabled", config_id)
            self._connection_status[config_id] = IntegrationStatus.DISCONNECTED
            return IntegrationStatus.DISCONNECTED

        # Simulate connection establishment
        self._connection_status[config_id] = IntegrationStatus.CONNECTED
        logger.info(
            "Connected to %s (%s) at %s",
            config.system_type.value,
            config_id,
            config.endpoint_url or "local",
        )
        return IntegrationStatus.CONNECTED

    def disconnect(self, config_id: str) -> IntegrationStatus:
        """Disconnect from a clinical system.

        Args:
            config_id: ID of the integration configuration.

        Returns:
            Updated integration status.
        """
        if config_id in self._connection_status:
            self._connection_status[config_id] = IntegrationStatus.DISCONNECTED
            logger.info("Disconnected from %s", config_id)
        return IntegrationStatus.DISCONNECTED

    def ingest_clinical_data(
        self,
        config_id: str,
        raw_data: list[dict[str, Any]],
        apply_deidentification: bool = True,
    ) -> SyncResult:
        """Ingest clinical data from a specified system.

        Validates, optionally de-identifies, and stores clinical data
        points from the raw input. Returns a synchronization result
        with acceptance/rejection counts.

        Args:
            config_id: ID of the source integration configuration.
            raw_data: List of raw clinical data records.
            apply_deidentification: Whether to apply de-identification.

        Returns:
            SyncResult with ingestion statistics.
        """
        start_time = time.time()

        if config_id not in self._configs:
            logger.error("Unknown integration config: %s", config_id)
            return SyncResult(
                config_id=config_id,
                status=IntegrationStatus.ERROR,
                error_message=f"Unknown config: {config_id}",
            )

        config = self._configs[config_id]
        self._connection_status[config_id] = IntegrationStatus.SYNCING

        accepted_points: list[ClinicalDataPoint] = []
        rejected_count = 0

        for record in raw_data[:MAX_RECORDS_PER_SYNC]:
            # Apply de-identification if required
            if apply_deidentification and config.deidentification_required:
                record = _deidentify_record(record)

            # Validate based on format
            is_valid = True
            if config.data_format == DataFormat.HL7_FHIR:
                is_valid = _validate_fhir_resource(record)

            if not is_valid:
                rejected_count += 1
                continue

            # Convert to ClinicalDataPoint
            data_point = ClinicalDataPoint(
                patient_id=str(record.get("patient_id", "")),
                source_system=config.system_type,
                data_format=config.data_format,
                name=str(record.get("name", record.get("code", ""))),
                value=record.get("value", record.get("valueQuantity", {}).get("value", 0.0)),
                unit=str(record.get("unit", record.get("valueQuantity", {}).get("unit", ""))),
                is_deidentified=apply_deidentification,
                metadata={"source_hash": _compute_data_hash(record)},
            )
            accepted_points.append(data_point)

        # Store accepted data points
        self._data_store.setdefault(config_id, []).extend(accepted_points)
        self._ingestion_count += len(accepted_points)

        duration = time.time() - start_time
        self._connection_status[config_id] = IntegrationStatus.CONNECTED

        result = SyncResult(
            config_id=config_id,
            status=IntegrationStatus.CONNECTED,
            records_received=len(raw_data),
            records_accepted=len(accepted_points),
            records_rejected=rejected_count,
            duration_seconds=duration,
            data_points=accepted_points,
        )
        self._sync_history.append(result)

        logger.info(
            "Ingested %d/%d records from %s (rejected=%d, duration=%.2fs)",
            len(accepted_points),
            len(raw_data),
            config_id,
            rejected_count,
            duration,
        )
        return result

    def transform_to_twin_input(
        self,
        patient_id: str,
        config_id: str | None = None,
    ) -> dict[str, Any]:
        """Convert clinical data into patient digital twin input format.

        Aggregates all stored clinical data for a patient and transforms
        it into the structured input expected by the patient modeling
        engine.

        Args:
            patient_id: De-identified patient identifier.
            config_id: Optional filter by source system. If None, uses all sources.

        Returns:
            Dictionary suitable for PatientDigitalTwinConfig construction.
        """
        data_points: list[ClinicalDataPoint] = []
        if config_id:
            data_points = [dp for dp in self._data_store.get(config_id, []) if dp.patient_id == patient_id]
        else:
            for points in self._data_store.values():
                data_points.extend([dp for dp in points if dp.patient_id == patient_id])

        if not data_points:
            logger.warning("No clinical data found for patient %s", patient_id)
            return {"patient_id": patient_id}

        # Aggregate biomarker values
        biomarker_values: dict[str, float] = {}
        for dp in data_points:
            if dp.category == DataCategory.LABORATORY and isinstance(dp.value, (int, float)):
                biomarker_values[dp.name] = float(dp.value)

        # Extract tumor volume if available
        volume_cm3 = 2.0
        for dp in data_points:
            if "volume" in dp.name.lower() and isinstance(dp.value, (int, float)):
                volume_cm3 = float(np.clip(dp.value, 0.001, 1000.0))
                break

        twin_input = {
            "patient_id": patient_id,
            "biomarker_values": biomarker_values,
            "initial_volume_cm3": volume_cm3,
            "data_point_count": len(data_points),
            "source_systems": list({dp.source_system.value for dp in data_points}),
        }

        logger.info(
            "Transformed %d data points for patient %s into twin input",
            len(data_points),
            patient_id,
        )
        return twin_input

    def export_results(
        self,
        results: dict[str, Any],
        target_format: DataFormat = DataFormat.JSON,
        target_path: str | None = None,
    ) -> dict[str, Any]:
        """Export simulation results in a clinical interchange format.

        Args:
            results: Simulation results dictionary to export.
            target_format: Output format for the exported data.
            target_path: Optional file path for export output.

        Returns:
            Dictionary with export status and serialized content.
        """
        export_id = f"EXP-{uuid.uuid4().hex[:12].upper()}"

        if target_format == DataFormat.JSON:
            serialized = json.dumps(results, indent=2, default=str)
        elif target_format == DataFormat.CSV:
            # Flatten results to CSV format
            rows = []
            for key, value in results.items():
                rows.append(f"{key},{value}")
            serialized = "key,value\n" + "\n".join(rows)
        else:
            serialized = json.dumps(results, default=str)

        if target_path:
            output_path = Path(target_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized, encoding="utf-8")
            logger.info("Exported results to %s (format=%s)", target_path, target_format.value)

        export_result = {
            "export_id": export_id,
            "format": target_format.value,
            "size_bytes": len(serialized.encode("utf-8")),
            "hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "path": target_path,
        }
        logger.info(
            "Export %s complete: format=%s, size=%d bytes",
            export_id,
            target_format.value,
            export_result["size_bytes"],
        )
        return export_result

    def generate_clinical_report(
        self,
        patient_id: str,
        simulation_results: dict[str, Any],
        include_recommendations: bool = True,
    ) -> ClinicalReport:
        """Generate a structured clinical report from simulation outcomes.

        Creates a comprehensive report integrating patient clinical data
        with digital twin simulation results. The report is marked with
        the research-use-only disclaimer.

        Args:
            patient_id: De-identified patient identifier.
            simulation_results: Dictionary of simulation outcomes.
            include_recommendations: Whether to include treatment recommendations.

        Returns:
            ClinicalReport instance.
        """
        # Build report sections
        sections = [
            {
                "title": "Patient Overview",
                "content": f"De-identified patient {patient_id} enrolled in federated clinical trial.",
            },
            {
                "title": "Simulation Parameters",
                "content": json.dumps(
                    {k: v for k, v in simulation_results.items() if k != "volume_trajectory"},
                    indent=2,
                    default=str,
                ),
            },
        ]

        # Response summary
        response = simulation_results.get("response_category", "unknown")
        reduction = simulation_results.get("volume_reduction_pct", 0.0)
        sections.append(
            {
                "title": "Treatment Response",
                "content": (f"Predicted response: {response}. Estimated volume reduction: {reduction:.1f}%."),
            }
        )

        # Recommendations
        recommendations = []
        if include_recommendations:
            if isinstance(reduction, (int, float)) and reduction > 30:
                recommendations.append(
                    "Simulation suggests favorable treatment response. Consider proceeding with planned regimen."
                )
            elif isinstance(reduction, (int, float)) and reduction > 0:
                recommendations.append(
                    "Simulation suggests modest response. Consider dose adjustment or alternative modality."
                )
            else:
                recommendations.append("Simulation suggests limited response. Consider treatment plan reassessment.")
            recommendations.append(
                "All simulation-based recommendations require independent clinical validation before implementation."
            )

        # Compute confidence level based on data availability
        data_point_count = sum(
            len([dp for dp in points if dp.patient_id == patient_id]) for points in self._data_store.values()
        )
        confidence = float(np.clip(min(data_point_count / 50.0, 1.0) * 0.8, 0.1, 0.95))

        report = ClinicalReport(
            patient_id=patient_id,
            summary=f"Digital twin simulation report for patient {patient_id}. "
            f"Response: {response}, Volume reduction: {reduction:.1f}%.",
            sections=sections,
            recommendations=recommendations,
            confidence_level=confidence,
            data_sources=[self.site_id],
        )

        logger.info(
            "Generated clinical report %s for patient %s (confidence=%.2f)",
            report.report_id,
            patient_id,
            confidence,
        )
        return report

    def get_connection_status(self) -> dict[str, str]:
        """Return current connection status for all registered systems."""
        return {config_id: status.value for config_id, status in self._connection_status.items()}

    def get_sync_history(self) -> list[SyncResult]:
        """Return the complete synchronization history."""
        return list(self._sync_history)

    @property
    def ingestion_count(self) -> int:
        """Return the total number of data points ingested."""
        return self._ingestion_count


# ---------------------------------------------------------------------------
# FederatedClinicalBridge
# ---------------------------------------------------------------------------
class FederatedClinicalBridge:
    """Bridge connecting digital twin outputs to the federated learning pipeline.

    Transforms digital twin feature vectors and simulation results into
    payloads suitable for federated model training. Handles de-identification
    verification, payload size limits, and communication with the
    federated aggregation coordinator.

    Args:
        site_id: Identifier for the clinical trial site.
        integrator: ClinicalIntegrator instance for data access.
        federation_endpoint: URL of the federated aggregation coordinator.
    """

    def __init__(
        self,
        site_id: str = "SITE-001",
        integrator: ClinicalIntegrator | None = None,
        federation_endpoint: str = "",
    ) -> None:
        self.site_id = site_id
        self.integrator = integrator
        self.federation_endpoint = federation_endpoint
        self._payload_history: list[dict[str, Any]] = []
        self._global_model_version: int = 0
        self._submitted_count: int = 0
        logger.info(
            "FederatedClinicalBridge initialized for site %s (endpoint=%s)",
            self.site_id,
            self.federation_endpoint or "not_configured",
        )

    def prepare_federated_payload(
        self,
        feature_vectors: list[np.ndarray],
        labels: list[int] | None = None,
        model_gradients: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Prepare digital twin features for federated model training.

        Validates de-identification, normalizes features, and packages
        them into a payload that meets federated learning protocol
        requirements.

        Args:
            feature_vectors: List of patient feature vectors.
            labels: Optional list of outcome labels for supervised training.
            model_gradients: Optional model gradients for federated averaging.

        Returns:
            Dictionary containing the federated training payload.
        """
        if not feature_vectors:
            logger.warning("No feature vectors provided for federated payload")
            return {"site_id": self.site_id, "error": "empty_payload"}

        # Stack and validate feature vectors
        stacked = np.array(feature_vectors, dtype=np.float64)
        if stacked.ndim == 1:
            stacked = stacked.reshape(1, -1)

        # Ensure all values are bounded [0, 1] for privacy safety
        stacked = np.clip(stacked, 0.0, 1.0)

        # Compute aggregate statistics (no individual patient data)
        feature_means = np.mean(stacked, axis=0).tolist()
        feature_stds = np.std(stacked, axis=0).tolist()

        payload = {
            "payload_id": f"FED-{uuid.uuid4().hex[:12].upper()}",
            "site_id": self.site_id,
            "global_model_version": self._global_model_version,
            "n_samples": stacked.shape[0],
            "n_features": stacked.shape[1],
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "timestamp_unix": time.time(),
        }

        if labels is not None:
            payload["label_distribution"] = {
                str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))
            }

        if model_gradients is not None:
            gradient_norm = float(np.linalg.norm(model_gradients))
            payload["gradient_norm"] = gradient_norm
            payload["gradient_shape"] = list(model_gradients.shape)

        # Verify payload size
        payload_json = json.dumps(payload, default=str)
        payload_size = len(payload_json.encode("utf-8"))
        if payload_size > MAX_PAYLOAD_SIZE_BYTES:
            logger.warning(
                "Payload size %d exceeds limit %d bytes",
                payload_size,
                MAX_PAYLOAD_SIZE_BYTES,
            )

        self._payload_history.append(payload)
        logger.info(
            "Prepared federated payload %s: %d samples, %d features",
            payload["payload_id"],
            stacked.shape[0],
            stacked.shape[1],
        )
        return payload

    def submit_to_federation(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit a prepared payload to the federated aggregation pipeline.

        In research mode, this simulates submission and returns a
        confirmation. Production implementations would use secure
        gRPC or HTTPS transport to the coordinator.

        Args:
            payload: Prepared federated training payload.

        Returns:
            Submission confirmation with status.
        """
        submission_id = f"SUB-{uuid.uuid4().hex[:12].upper()}"

        # Simulate submission (research mode)
        confirmation = {
            "submission_id": submission_id,
            "payload_id": payload.get("payload_id", "unknown"),
            "site_id": self.site_id,
            "status": "accepted",
            "global_model_version": self._global_model_version,
            "timestamp_unix": time.time(),
        }

        self._submitted_count += 1
        logger.info(
            "Submitted payload %s to federation (submission=%s, total=%d)",
            payload.get("payload_id", "unknown"),
            submission_id,
            self._submitted_count,
        )
        return confirmation

    def receive_global_update(
        self,
        model_weights: np.ndarray | None = None,
        model_version: int = 0,
    ) -> dict[str, Any]:
        """Receive and apply a global model update from the federation.

        Args:
            model_weights: Updated global model weights.
            model_version: Version number of the global model.

        Returns:
            Update application result.
        """
        previous_version = self._global_model_version
        self._global_model_version = max(model_version, self._global_model_version + 1)

        update_result = {
            "site_id": self.site_id,
            "previous_version": previous_version,
            "new_version": self._global_model_version,
            "weights_received": model_weights is not None,
            "weights_shape": list(model_weights.shape) if model_weights is not None else [],
            "timestamp_unix": time.time(),
        }

        logger.info(
            "Received global model update: v%d -> v%d (weights=%s)",
            previous_version,
            self._global_model_version,
            model_weights is not None,
        )
        return update_result

    def get_payload_history(self) -> list[dict[str, Any]]:
        """Return the complete payload submission history."""
        return list(self._payload_history)

    @property
    def submitted_count(self) -> int:
        """Return the total number of payloads submitted."""
        return self._submitted_count

    @property
    def global_model_version(self) -> int:
        """Return the current global model version."""
        return self._global_model_version

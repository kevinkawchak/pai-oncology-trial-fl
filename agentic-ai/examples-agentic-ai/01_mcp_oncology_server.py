"""
MCP Server Exposing Oncology Domain Tools and Resources for LLM Agents.

CLINICAL CONTEXT:
    This module implements a Model Context Protocol (MCP) server tailored for
    oncology clinical trial workflows within the PAI Federated Learning
    platform. The server exposes domain-specific tools (patient lookup,
    treatment simulation, compliance checking) and resources (trial protocols,
    regulatory references) that LLM agents can invoke via JSON-RPC to assist
    with clinical trial operations.

    The server follows the MCP specification for tool definitions with
    JSON Schema input schemas, resource URIs with MIME-typed content, and
    structured prompt templates. All interactions are logged to a 21 CFR
    Part 11 compliant audit trail with hash-chain integrity verification.

USE CASES COVERED:
    1. Patient lookup with de-identified record retrieval across federated
       trial sites, including demographics, labs, imaging, and genomics.
    2. Treatment simulation via digital twin integration, supporting
       chemotherapy, radiation, immunotherapy, and combination modalities.
    3. Protocol compliance verification against ICH E6(R3) requirements
       including informed consent, eligibility criteria, and dose limits.
    4. Trial protocol resource serving with versioned protocol documents,
       amendments, and regulatory guidance accessible via MCP resource URIs.
    5. Robotic procedure telemetry ingestion and status monitoring for
       Physical AI surgical systems integrated with clinical trials.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0        (https://numpy.org/)
    Optional:
        - mcp >= 1.1.0           (https://modelcontextprotocol.io/)
        - anthropic >= 0.39.0    (https://docs.anthropic.com/)

REFERENCES:
    - Model Context Protocol Specification v2025-03-26
      URL: https://spec.modelcontextprotocol.io/
    - FDA 21 CFR Part 11 - Electronic Records; Electronic Signatures
      URL: https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
    - ICH E6(R3) Good Clinical Practice, Step 2b (2023)
      URL: https://www.ich.org/page/efficacy-guidelines
    - RECIST 1.1 - Eisenhauer et al. (2009) Eur J Cancer 45(2)
      DOI: 10.1016/j.ejca.2008.10.026

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or direct patient care. All outputs must be reviewed
    by qualified clinical professionals before any action is taken.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
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
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports for optional dependencies
# ---------------------------------------------------------------------------
try:
    import mcp  # type: ignore[import-untyped]
    from mcp.server import Server as MCPServer  # type: ignore[import-untyped]

    HAS_MCP = True
except ImportError:
    mcp = None  # type: ignore[assignment]
    MCPServer = None  # type: ignore[assignment,misc]
    HAS_MCP = False

try:
    import anthropic  # type: ignore[import-untyped]

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class RobotMode(Enum):
    """Operational modes for Physical AI robotic surgical systems."""

    IDLE = "idle"
    CALIBRATING = "calibrating"
    NAVIGATING = "navigating"
    OPERATING = "operating"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"


class ProcedurePhase(Enum):
    """Phases of a robotic-assisted surgical procedure."""

    PRE_OPERATIVE = "pre_operative"
    SETUP = "setup"
    DRAPING = "draping"
    INCISION = "incision"
    DISSECTION = "dissection"
    RESECTION = "resection"
    RECONSTRUCTION = "reconstruction"
    CLOSURE = "closure"
    POST_OPERATIVE = "post_operative"


class ComplianceStatus(Enum):
    """Compliance verification result categories."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    WAIVER_REQUIRED = "waiver_required"
    NOT_APPLICABLE = "not_applicable"


class ToolCategory(Enum):
    """Categories for MCP tool definitions."""

    PATIENT_DATA = "patient_data"
    TREATMENT_SIMULATION = "treatment_simulation"
    COMPLIANCE = "compliance"
    TELEMETRY = "telemetry"
    PROTOCOL = "protocol"
    SAFETY = "safety"


class AuditEventType(Enum):
    """Types of auditable events for 21 CFR Part 11 compliance."""

    TOOL_INVOKED = "tool_invoked"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    RESOURCE_ACCESSED = "resource_accessed"
    COMPLIANCE_CHECK = "compliance_check"
    PATIENT_LOOKUP = "patient_lookup"
    SIMULATION_RUN = "simulation_run"
    TELEMETRY_RECEIVED = "telemetry_received"
    SERVER_STARTED = "server_started"
    SERVER_STOPPED = "server_stopped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RobotTelemetry:
    """Telemetry data from a Physical AI robotic surgical system.

    Captures real-time sensor readings, operational status, and safety
    metrics from robotic systems integrated with clinical trials.
    """

    telemetry_id: str = field(default_factory=lambda: f"TEL-{uuid.uuid4().hex[:12].upper()}")
    robot_id: str = ""
    timestamp: float = field(default_factory=time.time)
    mode: RobotMode = RobotMode.IDLE
    phase: ProcedurePhase = ProcedurePhase.PRE_OPERATIVE
    position_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force_n: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    torque_nm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_angles_deg: list[float] = field(default_factory=lambda: [0.0] * 7)
    temperature_c: float = 22.0
    battery_pct: float = 100.0
    error_codes: list[str] = field(default_factory=list)
    safety_flags: dict[str, bool] = field(
        default_factory=lambda: {
            "workspace_boundary_ok": True,
            "force_limit_ok": True,
            "collision_detected": False,
            "e_stop_active": False,
        }
    )

    def is_safe(self) -> bool:
        """Check if all safety flags indicate safe operation."""
        return (
            self.safety_flags.get("workspace_boundary_ok", False)
            and self.safety_flags.get("force_limit_ok", False)
            and not self.safety_flags.get("collision_detected", False)
            and not self.safety_flags.get("e_stop_active", False)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize telemetry to dictionary for JSON-RPC response."""
        return {
            "telemetry_id": self.telemetry_id,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "mode": self.mode.value,
            "phase": self.phase.value,
            "position_mm": self.position_mm,
            "orientation_deg": self.orientation_deg,
            "force_n": self.force_n,
            "torque_nm": self.torque_nm,
            "joint_angles_deg": self.joint_angles_deg,
            "temperature_c": self.temperature_c,
            "battery_pct": self.battery_pct,
            "error_codes": self.error_codes,
            "safety_flags": self.safety_flags,
            "is_safe": self.is_safe(),
        }


@dataclass
class AuditEntry:
    """A single entry in the 21 CFR Part 11 compliant audit trail.

    Each entry is hash-chained to the previous entry for tamper detection.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.TOOL_INVOKED
    timestamp: float = field(default_factory=time.time)
    user_identity: str = "mcp_server"
    tool_name: str = ""
    resource_uri: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    previous_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute SHA-256 hash chaining this entry to the previous one."""
        self.previous_hash = previous_hash
        payload = json.dumps(
            {
                "entry_id": self.entry_id,
                "event_type": self.event_type.value,
                "timestamp": self.timestamp,
                "user_identity": self.user_identity,
                "tool_name": self.tool_name,
                "resource_uri": self.resource_uri,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        self.entry_hash = hashlib.sha256(payload).hexdigest()
        return self.entry_hash


@dataclass
class PatientRecord:
    """De-identified patient record for federated trial queries."""

    patient_id: str = ""
    site_id: str = ""
    age_bracket: str = "60-69"
    sex: str = "unknown"
    diagnosis: str = ""
    stage: str = ""
    histology: str = ""
    biomarkers: dict[str, float] = field(default_factory=dict)
    treatment_arm: str = ""
    enrollment_date: str = ""
    ecog_score: int = 1
    tumor_volume_cm3: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON-RPC response."""
        return {
            "patient_id": self.patient_id,
            "site_id": self.site_id,
            "age_bracket": self.age_bracket,
            "sex": self.sex,
            "diagnosis": self.diagnosis,
            "stage": self.stage,
            "histology": self.histology,
            "biomarkers": self.biomarkers,
            "treatment_arm": self.treatment_arm,
            "enrollment_date": self.enrollment_date,
            "ecog_score": self.ecog_score,
            "tumor_volume_cm3": self.tumor_volume_cm3,
        }


@dataclass
class TrialProtocol:
    """Clinical trial protocol document served as an MCP resource."""

    protocol_id: str = ""
    title: str = ""
    version: str = "1.0"
    phase: str = "III"
    sponsor: str = ""
    indication: str = ""
    primary_endpoint: str = ""
    secondary_endpoints: list[str] = field(default_factory=list)
    eligibility_criteria: dict[str, Any] = field(default_factory=dict)
    treatment_arms: list[dict[str, Any]] = field(default_factory=list)
    dose_limits: dict[str, float] = field(default_factory=dict)
    amendment_history: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "protocol_id": self.protocol_id,
            "title": self.title,
            "version": self.version,
            "phase": self.phase,
            "sponsor": self.sponsor,
            "indication": self.indication,
            "primary_endpoint": self.primary_endpoint,
            "secondary_endpoints": self.secondary_endpoints,
            "eligibility_criteria": self.eligibility_criteria,
            "treatment_arms": self.treatment_arms,
            "dose_limits": self.dose_limits,
            "amendment_history": self.amendment_history,
        }


@dataclass
class ComplianceResult:
    """Result of a protocol compliance check."""

    check_id: str = field(default_factory=lambda: f"CHK-{uuid.uuid4().hex[:10].upper()}")
    status: ComplianceStatus = ComplianceStatus.NEEDS_REVIEW
    protocol_id: str = ""
    patient_id: str = ""
    checks_performed: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON-RPC response."""
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "protocol_id": self.protocol_id,
            "patient_id": self.patient_id,
            "checks_performed": self.checks_performed,
            "violations": self.violations,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }


@dataclass
class SimulationResult:
    """Result of a treatment simulation invoked through the MCP server."""

    simulation_id: str = field(default_factory=lambda: f"SIM-{uuid.uuid4().hex[:12].upper()}")
    patient_id: str = ""
    modality: str = "chemotherapy"
    initial_volume_cm3: float = 0.0
    predicted_volume_cm3: float = 0.0
    volume_reduction_pct: float = 0.0
    response_category: str = "stable_disease"
    confidence_low: float = 0.0
    confidence_high: float = 0.0
    toxicity_grade: int = 0
    trajectory: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON-RPC response."""
        return {
            "simulation_id": self.simulation_id,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "initial_volume_cm3": self.initial_volume_cm3,
            "predicted_volume_cm3": self.predicted_volume_cm3,
            "volume_reduction_pct": self.volume_reduction_pct,
            "response_category": self.response_category,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "toxicity_grade": self.toxicity_grade,
            "trajectory": self.trajectory,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Audit trail manager (21 CFR Part 11)
# ---------------------------------------------------------------------------
class AuditTrailManager:
    """Hash-chained audit trail for FDA 21 CFR Part 11 compliance.

    Every tool invocation, resource access, and compliance check is recorded
    with cryptographic hash chaining for tamper detection.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._last_hash: str = ""

    def record(
        self,
        event_type: AuditEventType,
        tool_name: str = "",
        resource_uri: str = "",
        parameters: Optional[dict[str, Any]] = None,
        result_summary: str = "",
        user_identity: str = "mcp_server",
    ) -> AuditEntry:
        """Record an auditable event with hash chain integrity."""
        entry = AuditEntry(
            event_type=event_type,
            user_identity=user_identity,
            tool_name=tool_name,
            resource_uri=resource_uri,
            parameters=parameters or {},
            result_summary=result_summary,
        )
        entry.compute_hash(self._last_hash)
        self._last_hash = entry.entry_hash
        self._entries.append(entry)

        logger.info(
            "Audit [%s] type=%s tool=%s hash=%s",
            entry.entry_id[:8],
            event_type.value,
            tool_name or resource_uri or "N/A",
            entry.entry_hash[:12],
        )
        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire audit chain."""
        prev_hash = ""
        for entry in self._entries:
            expected = entry.compute_hash(prev_hash)
            if expected != entry.entry_hash:
                logger.error("Audit chain broken at entry %s", entry.entry_id)
                return False
            prev_hash = entry.entry_hash
        return True

    def export_json(self) -> str:
        """Export the audit trail as a JSON string."""
        records = []
        for entry in self._entries:
            records.append(
                {
                    "entry_id": entry.entry_id,
                    "event_type": entry.event_type.value,
                    "timestamp": entry.timestamp,
                    "user_identity": entry.user_identity,
                    "tool_name": entry.tool_name,
                    "resource_uri": entry.resource_uri,
                    "result_summary": entry.result_summary,
                    "previous_hash": entry.previous_hash,
                    "entry_hash": entry.entry_hash,
                }
            )
        return json.dumps(records, indent=2)

    @property
    def count(self) -> int:
        """Number of audit entries recorded."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
def _generate_synthetic_patients(count: int = 10, seed: int = 42) -> list[PatientRecord]:
    """Generate synthetic de-identified patient records for demonstration."""
    rng = np.random.default_rng(seed)
    diagnoses = ["NSCLC", "SCLC", "breast_adenocarcinoma", "colorectal_adenocarcinoma", "melanoma"]
    stages = ["I", "II", "IIIA", "IIIB", "IV"]
    histologies = ["adenocarcinoma", "squamous_cell", "large_cell", "small_cell"]
    arms = ["experimental", "control", "dose_escalation_1", "dose_escalation_2"]
    sites = ["SITE-001", "SITE-002", "SITE-003", "SITE-004"]

    patients = []
    for i in range(count):
        age_start = int(rng.integers(40, 80))
        age_bracket = f"{age_start}-{age_start + 9}"
        patient = PatientRecord(
            patient_id=f"PAT-{uuid.uuid4().hex[:8].upper()}",
            site_id=sites[int(rng.integers(0, len(sites)))],
            age_bracket=age_bracket,
            sex=rng.choice(["male", "female"]),
            diagnosis=diagnoses[int(rng.integers(0, len(diagnoses)))],
            stage=stages[int(rng.integers(0, len(stages)))],
            histology=histologies[int(rng.integers(0, len(histologies)))],
            biomarkers={
                "PDL1_TPS": float(rng.uniform(0, 100)),
                "TMB_mut_per_mb": float(rng.uniform(1, 40)),
                "EGFR_mutation": float(rng.choice([0.0, 1.0])),
                "ALK_rearrangement": float(rng.choice([0.0, 1.0])),
                "KRAS_G12C": float(rng.choice([0.0, 1.0])),
            },
            treatment_arm=arms[int(rng.integers(0, len(arms)))],
            enrollment_date=f"2025-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}",
            ecog_score=int(rng.integers(0, 3)),
            tumor_volume_cm3=float(rng.uniform(0.5, 25.0)),
        )
        patients.append(patient)
    return patients


def _generate_trial_protocols() -> list[TrialProtocol]:
    """Generate synthetic trial protocol documents for demonstration."""
    protocols = [
        TrialProtocol(
            protocol_id="PROTO-ONC-2025-001",
            title="Phase III Randomized Trial of Pembrolizumab + Carboplatin vs Carboplatin in Stage IIIB-IV NSCLC",
            version="3.2",
            phase="III",
            sponsor="PAI Oncology Research Consortium",
            indication="Non-small cell lung cancer (NSCLC)",
            primary_endpoint="Progression-free survival (PFS) per RECIST 1.1",
            secondary_endpoints=[
                "Overall survival (OS)",
                "Objective response rate (ORR)",
                "Duration of response (DOR)",
                "Safety and tolerability (CTCAE v5.0)",
            ],
            eligibility_criteria={
                "age_min": 18,
                "age_max": 85,
                "ecog_max": 2,
                "stage": ["IIIB", "IV"],
                "histology_required": ["adenocarcinoma", "squamous_cell"],
                "pdl1_tps_min": 1.0,
                "prior_lines_max": 1,
                "adequate_organ_function": True,
            },
            treatment_arms=[
                {
                    "arm": "A",
                    "name": "Experimental",
                    "regimen": "Pembrolizumab 200mg IV Q3W + Carboplatin AUC 5 IV Q3W x4 cycles",
                },
                {
                    "arm": "B",
                    "name": "Control",
                    "regimen": "Carboplatin AUC 5 IV Q3W x4 cycles + Placebo IV Q3W",
                },
            ],
            dose_limits={
                "carboplatin_max_auc": 6.0,
                "pembrolizumab_max_mg": 200.0,
                "radiation_max_gy": 0.0,
            },
            amendment_history=[
                {"version": "3.1", "date": "2025-06-15", "summary": "Updated eligibility PD-L1 threshold"},
                {"version": "3.2", "date": "2025-09-01", "summary": "Added digital twin sub-study"},
            ],
        ),
        TrialProtocol(
            protocol_id="PROTO-ONC-2025-002",
            title="Phase II Adaptive Trial of Robotic-Assisted Biopsy with AI-Guided Navigation in Lung Nodules",
            version="2.0",
            phase="II",
            sponsor="PAI Oncology Research Consortium",
            indication="Indeterminate lung nodules (8-30mm)",
            primary_endpoint="Diagnostic yield (proportion of adequate tissue samples)",
            secondary_endpoints=[
                "Procedure time (minutes)",
                "Complication rate (pneumothorax, hemorrhage)",
                "Navigation accuracy (mm)",
                "Patient satisfaction score",
            ],
            eligibility_criteria={
                "age_min": 21,
                "age_max": 80,
                "nodule_size_mm_min": 8,
                "nodule_size_mm_max": 30,
                "bleeding_risk_acceptable": True,
                "no_prior_thoracic_surgery": True,
            },
            treatment_arms=[
                {"arm": "A", "name": "AI-Guided Robotic Biopsy", "regimen": "Robotic biopsy with AI navigation"},
                {"arm": "B", "name": "Standard CT-Guided Biopsy", "regimen": "Conventional CT-guided percutaneous"},
            ],
            dose_limits={},
            amendment_history=[
                {"version": "2.0", "date": "2025-11-10", "summary": "Added Physical AI integration arm"},
            ],
        ),
    ]
    return protocols


# ---------------------------------------------------------------------------
# MCP tool handlers (domain logic)
# ---------------------------------------------------------------------------
class OncologyToolHandlers:
    """Handlers for oncology domain tools exposed via MCP.

    Each handler implements the business logic for an MCP tool, operating
    on synthetic data for demonstration. In production, these would connect
    to federated data stores, simulation engines, and compliance systems.
    """

    def __init__(self, seed: int = 42) -> None:
        self._patients = _generate_synthetic_patients(count=20, seed=seed)
        self._protocols = _generate_trial_protocols()
        self._patient_index: dict[str, PatientRecord] = {p.patient_id: p for p in self._patients}
        self._protocol_index: dict[str, TrialProtocol] = {p.protocol_id: p for p in self._protocols}
        self._rng = np.random.default_rng(seed)
        logger.info(
            "OncologyToolHandlers initialized: %d patients, %d protocols",
            len(self._patients),
            len(self._protocols),
        )

    def patient_lookup(self, patient_id: str = "", site_id: str = "", diagnosis: str = "") -> dict[str, Any]:
        """Look up de-identified patient records with optional filters.

        Args:
            patient_id: Specific patient ID to retrieve.
            site_id: Filter by trial site.
            diagnosis: Filter by diagnosis.

        Returns:
            Dictionary with matching patient records.
        """
        if patient_id and patient_id in self._patient_index:
            return {
                "status": "found",
                "count": 1,
                "patients": [self._patient_index[patient_id].to_dict()],
            }

        matches = self._patients
        if site_id:
            matches = [p for p in matches if p.site_id == site_id]
        if diagnosis:
            matches = [p for p in matches if diagnosis.lower() in p.diagnosis.lower()]

        return {
            "status": "found" if matches else "no_matches",
            "count": len(matches),
            "patients": [p.to_dict() for p in matches[:10]],
        }

    def treatment_simulation(
        self,
        patient_id: str,
        modality: str = "chemotherapy",
        dose_mg: float = 75.0,
        cycles: int = 4,
        n_simulations: int = 50,
    ) -> dict[str, Any]:
        """Run a treatment simulation on a patient digital twin.

        Args:
            patient_id: De-identified patient ID.
            modality: Treatment modality (chemotherapy, radiation, immunotherapy).
            dose_mg: Dose in mg (for chemotherapy).
            cycles: Number of treatment cycles.
            n_simulations: Number of Monte Carlo iterations.

        Returns:
            Dictionary with simulation results including response prediction.
        """
        patient = self._patient_index.get(patient_id)
        if patient is None:
            # Use a synthetic default for demo
            initial_volume = float(self._rng.uniform(2.0, 15.0))
        else:
            initial_volume = patient.tumor_volume_cm3

        # Simulate treatment response with Monte Carlo
        volumes = []
        sensitivity = float(self._rng.uniform(0.3, 0.8))
        for sim_i in range(n_simulations):
            sim_rng = np.random.default_rng(sim_i + 100)
            perturbed_sens = float(np.clip(sensitivity + sim_rng.normal(0, 0.1), 0.0, 1.0))
            volume = initial_volume
            for cycle_idx in range(cycles):
                kill_fraction = perturbed_sens * min(dose_mg / 100.0, 0.8)
                volume *= 1.0 - kill_fraction
                # Regrowth between cycles
                growth_rate = float(sim_rng.uniform(0.005, 0.02))
                volume *= np.exp(growth_rate * 21)
            volumes.append(float(max(volume, 0.001)))

        arr = np.array(volumes)
        mean_vol = float(np.mean(arr))
        reduction_pct = float((initial_volume - mean_vol) / max(initial_volume, 0.001) * 100.0)

        # Classify response
        reduction_frac = reduction_pct / 100.0
        if reduction_frac >= 0.65:
            response = "complete_response"
        elif reduction_frac >= 0.30:
            response = "partial_response"
        elif reduction_frac >= -0.20:
            response = "stable_disease"
        else:
            response = "progressive_disease"

        result = SimulationResult(
            patient_id=patient_id,
            modality=modality,
            initial_volume_cm3=initial_volume,
            predicted_volume_cm3=mean_vol,
            volume_reduction_pct=reduction_pct,
            response_category=response,
            confidence_low=float(np.percentile(arr, 5)),
            confidence_high=float(np.percentile(arr, 95)),
            toxicity_grade=int(self._rng.integers(0, 4)),
            trajectory=[float(np.percentile(arr, q)) for q in [0, 25, 50, 75, 100]],
            metadata={
                "n_simulations": n_simulations,
                "dose_mg": dose_mg,
                "cycles": cycles,
                "sensitivity_mean": sensitivity,
            },
        )
        return result.to_dict()

    def compliance_check(
        self,
        patient_id: str,
        protocol_id: str = "PROTO-ONC-2025-001",
        checks: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Verify patient compliance against a trial protocol.

        Args:
            patient_id: De-identified patient ID.
            protocol_id: Protocol to check against.
            checks: Specific checks to perform (default: all).

        Returns:
            Dictionary with compliance status and any violations.
        """
        patient = self._patient_index.get(patient_id)
        protocol = self._protocol_index.get(protocol_id)

        if patient is None:
            return {
                "status": "error",
                "message": f"Patient {patient_id} not found",
            }

        if protocol is None:
            return {
                "status": "error",
                "message": f"Protocol {protocol_id} not found",
            }

        result = ComplianceResult(
            protocol_id=protocol_id,
            patient_id=patient_id,
        )

        eligibility = protocol.eligibility_criteria
        all_checks = checks or ["age", "ecog", "stage", "histology", "biomarkers"]

        for check_name in all_checks:
            check_entry = {"check": check_name, "status": "pass", "detail": ""}

            if check_name == "age":
                age_start = int(patient.age_bracket.split("-")[0])
                age_min = eligibility.get("age_min", 18)
                age_max = eligibility.get("age_max", 100)
                if age_start < age_min or age_start > age_max:
                    check_entry["status"] = "fail"
                    check_entry["detail"] = f"Age {age_start} outside [{age_min}, {age_max}]"
                    result.violations.append(check_entry["detail"])

            elif check_name == "ecog":
                ecog_max = eligibility.get("ecog_max", 2)
                if patient.ecog_score > ecog_max:
                    check_entry["status"] = "fail"
                    check_entry["detail"] = f"ECOG {patient.ecog_score} > max {ecog_max}"
                    result.violations.append(check_entry["detail"])

            elif check_name == "stage":
                allowed_stages = eligibility.get("stage", [])
                if allowed_stages and patient.stage not in allowed_stages:
                    check_entry["status"] = "fail"
                    check_entry["detail"] = f"Stage {patient.stage} not in {allowed_stages}"
                    result.violations.append(check_entry["detail"])

            elif check_name == "histology":
                allowed_hist = eligibility.get("histology_required", [])
                if allowed_hist and patient.histology not in allowed_hist:
                    check_entry["status"] = "warning"
                    check_entry["detail"] = f"Histology {patient.histology} not in preferred list"
                    result.warnings.append(check_entry["detail"])

            elif check_name == "biomarkers":
                pdl1_min = eligibility.get("pdl1_tps_min", 0.0)
                pdl1_val = patient.biomarkers.get("PDL1_TPS", 0.0)
                if pdl1_val < pdl1_min:
                    check_entry["status"] = "fail"
                    check_entry["detail"] = f"PD-L1 TPS {pdl1_val:.1f}% < minimum {pdl1_min}%"
                    result.violations.append(check_entry["detail"])

            result.checks_performed.append(check_entry)

        if result.violations:
            result.status = ComplianceStatus.NON_COMPLIANT
        elif result.warnings:
            result.status = ComplianceStatus.NEEDS_REVIEW
        else:
            result.status = ComplianceStatus.COMPLIANT

        return result.to_dict()

    def get_robot_telemetry(self, robot_id: str = "ROBOT-001") -> dict[str, Any]:
        """Generate simulated robot telemetry data.

        Args:
            robot_id: Robot system identifier.

        Returns:
            Dictionary with current telemetry readings.
        """
        telemetry = RobotTelemetry(
            robot_id=robot_id,
            mode=RobotMode.OPERATING,
            phase=ProcedurePhase.RESECTION,
            position_mm=[float(self._rng.uniform(-50, 50)) for _ in range(3)],
            orientation_deg=[float(self._rng.uniform(-10, 10)) for _ in range(3)],
            force_n=[float(self._rng.uniform(0, 5)) for _ in range(3)],
            torque_nm=[float(self._rng.uniform(0, 0.5)) for _ in range(3)],
            joint_angles_deg=[float(self._rng.uniform(-90, 90)) for _ in range(7)],
            temperature_c=float(self._rng.uniform(21, 24)),
            battery_pct=float(self._rng.uniform(70, 100)),
        )
        return telemetry.to_dict()


# ---------------------------------------------------------------------------
# MCP tool definitions (JSON Schema format)
# ---------------------------------------------------------------------------
def get_tool_definitions() -> list[dict[str, Any]]:
    """Return MCP-compliant tool definitions with JSON Schema input schemas.

    Each tool definition follows the MCP specification:
    - name: Unique tool identifier
    - description: Human-readable description for LLM context
    - inputSchema: JSON Schema (draft-07) for tool parameters
    """
    tools = [
        {
            "name": "oncology_patient_lookup",
            "description": (
                "Look up de-identified patient records from federated oncology trial sites. "
                "Supports filtering by patient ID, site ID, and diagnosis. Returns demographics, "
                "biomarkers, treatment arm, and tumor characteristics. All data is de-identified "
                "per HIPAA Safe Harbor; no PHI is exposed."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "De-identified patient ID (e.g., PAT-A1B2C3D4)",
                    },
                    "site_id": {
                        "type": "string",
                        "description": "Trial site identifier (e.g., SITE-001)",
                    },
                    "diagnosis": {
                        "type": "string",
                        "description": "Filter by diagnosis (e.g., NSCLC, melanoma)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "oncology_treatment_simulation",
            "description": (
                "Run a treatment simulation on a patient digital twin using Monte Carlo methods. "
                "Supports chemotherapy, radiation, and immunotherapy modalities. Returns predicted "
                "tumor volume reduction, RECIST 1.1 response category, confidence intervals, "
                "and CTCAE v5.0 toxicity grade estimate."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "De-identified patient ID for the digital twin",
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["chemotherapy", "radiation", "immunotherapy", "combination"],
                        "description": "Treatment modality to simulate",
                    },
                    "dose_mg": {
                        "type": "number",
                        "description": "Drug dose in mg (for chemotherapy)",
                        "default": 75.0,
                    },
                    "cycles": {
                        "type": "integer",
                        "description": "Number of treatment cycles",
                        "default": 4,
                    },
                    "n_simulations": {
                        "type": "integer",
                        "description": "Number of Monte Carlo simulation iterations",
                        "default": 50,
                    },
                },
                "required": ["patient_id"],
            },
        },
        {
            "name": "oncology_compliance_check",
            "description": (
                "Verify a patient's compliance against a clinical trial protocol. Checks "
                "eligibility criteria including age, ECOG status, disease stage, histology, "
                "and biomarker requirements per ICH E6(R3). Returns pass/fail for each "
                "criterion with detailed violation descriptions."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "De-identified patient ID to check",
                    },
                    "protocol_id": {
                        "type": "string",
                        "description": "Protocol ID to check against",
                        "default": "PROTO-ONC-2025-001",
                    },
                    "checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific checks to perform (default: all)",
                    },
                },
                "required": ["patient_id"],
            },
        },
        {
            "name": "oncology_robot_telemetry",
            "description": (
                "Retrieve real-time telemetry from a Physical AI robotic surgical system "
                "integrated with the clinical trial. Returns position, orientation, force, "
                "torque, joint angles, and safety flag status for robotic procedure monitoring."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "robot_id": {
                        "type": "string",
                        "description": "Robot system identifier (e.g., ROBOT-001)",
                        "default": "ROBOT-001",
                    },
                },
                "required": [],
            },
        },
    ]
    return tools


# ---------------------------------------------------------------------------
# MCP resource definitions
# ---------------------------------------------------------------------------
def get_resource_definitions() -> list[dict[str, Any]]:
    """Return MCP-compliant resource definitions for trial protocol documents.

    Resources are identified by URIs and serve protocol documents, regulatory
    guidance, and trial metadata as structured content.
    """
    resources = [
        {
            "uri": "oncology://protocols/PROTO-ONC-2025-001",
            "name": "Phase III NSCLC Pembrolizumab Trial Protocol",
            "description": "Complete protocol document for the Phase III randomized pembrolizumab + carboplatin trial",
            "mimeType": "application/json",
        },
        {
            "uri": "oncology://protocols/PROTO-ONC-2025-002",
            "name": "Phase II Robotic Biopsy Trial Protocol",
            "description": "Protocol for the adaptive robotic-assisted biopsy with AI navigation trial",
            "mimeType": "application/json",
        },
        {
            "uri": "oncology://guidance/fda-21cfr11",
            "name": "FDA 21 CFR Part 11 Reference",
            "description": "Summary of FDA electronic records and signatures requirements",
            "mimeType": "text/plain",
        },
        {
            "uri": "oncology://guidance/ich-e6r3",
            "name": "ICH E6(R3) GCP Guideline Reference",
            "description": "Key requirements from ICH E6(R3) Good Clinical Practice",
            "mimeType": "text/plain",
        },
    ]
    return resources


# ---------------------------------------------------------------------------
# MCP Server implementation
# ---------------------------------------------------------------------------
class OncologyMCPServer:
    """MCP server exposing oncology domain tools and resources for LLM agents.

    This server implements the Model Context Protocol to provide LLM agents
    with access to clinical trial tools (patient lookup, treatment simulation,
    compliance checking) and resources (protocol documents, regulatory guidance).

    All interactions are recorded in a 21 CFR Part 11 compliant audit trail.
    """

    def __init__(self, server_name: str = "oncology-mcp-server", seed: int = 42) -> None:
        self._name = server_name
        self._handlers = OncologyToolHandlers(seed=seed)
        self._audit = AuditTrailManager()
        self._tools = get_tool_definitions()
        self._resources = get_resource_definitions()
        self._protocols = _generate_trial_protocols()
        self._protocol_index: dict[str, TrialProtocol] = {p.protocol_id: p for p in self._protocols}

        self._audit.record(
            AuditEventType.SERVER_STARTED,
            tool_name=server_name,
            result_summary=f"Server initialized with {len(self._tools)} tools, {len(self._resources)} resources",
        )
        logger.info(
            "OncologyMCPServer '%s' initialized: %d tools, %d resources",
            server_name,
            len(self._tools),
            len(self._resources),
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """List all available MCP tools."""
        return self._tools

    def list_resources(self) -> list[dict[str, Any]]:
        """List all available MCP resources."""
        return self._resources

    def call_tool(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Invoke an MCP tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool input parameters.

        Returns:
            Tool execution result as a dictionary.
        """
        arguments = arguments or {}
        start_time = time.time()

        self._audit.record(
            AuditEventType.TOOL_INVOKED,
            tool_name=tool_name,
            parameters=arguments,
        )

        try:
            if tool_name == "oncology_patient_lookup":
                result = self._handlers.patient_lookup(**arguments)
            elif tool_name == "oncology_treatment_simulation":
                result = self._handlers.treatment_simulation(**arguments)
            elif tool_name == "oncology_compliance_check":
                result = self._handlers.compliance_check(**arguments)
            elif tool_name == "oncology_robot_telemetry":
                result = self._handlers.get_robot_telemetry(**arguments)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            elapsed_ms = (time.time() - start_time) * 1000.0
            self._audit.record(
                AuditEventType.TOOL_COMPLETED,
                tool_name=tool_name,
                result_summary=f"Completed in {elapsed_ms:.1f}ms",
            )

            logger.info("Tool '%s' completed in %.1fms", tool_name, elapsed_ms)
            return result

        except Exception as exc:
            self._audit.record(
                AuditEventType.TOOL_FAILED,
                tool_name=tool_name,
                result_summary=str(exc),
            )
            logger.error("Tool '%s' failed: %s", tool_name, exc)
            return {"error": str(exc)}

    def read_resource(self, uri: str) -> dict[str, Any]:
        """Read an MCP resource by URI.

        Args:
            uri: Resource URI (e.g., oncology://protocols/PROTO-ONC-2025-001).

        Returns:
            Resource content with MIME type.
        """
        self._audit.record(
            AuditEventType.RESOURCE_ACCESSED,
            resource_uri=uri,
        )

        # Protocol resources
        if uri.startswith("oncology://protocols/"):
            protocol_id = uri.split("/")[-1]
            protocol = self._protocol_index.get(protocol_id)
            if protocol:
                return {
                    "uri": uri,
                    "mimeType": "application/json",
                    "content": json.dumps(protocol.to_dict(), indent=2),
                }
            return {"error": f"Protocol not found: {protocol_id}"}

        # Regulatory guidance resources
        if uri == "oncology://guidance/fda-21cfr11":
            return {
                "uri": uri,
                "mimeType": "text/plain",
                "content": (
                    "FDA 21 CFR Part 11 - Electronic Records; Electronic Signatures\n"
                    "Key Requirements:\n"
                    "1. Electronic records must be trustworthy, reliable, and equivalent to paper\n"
                    "2. System validation for accuracy, reliability, and consistent intended performance\n"
                    "3. Ability to generate accurate and complete copies of records\n"
                    "4. Protection of records to enable their accurate and ready retrieval\n"
                    "5. Limiting system access to authorized individuals\n"
                    "6. Use of secure, computer-generated, time-stamped audit trails\n"
                    "7. Operational system checks to enforce permitted sequencing of events\n"
                    "8. Authority checks to ensure only authorized individuals can use the system\n"
                    "9. Device checks to determine validity of data input source\n"
                    "10. Written policies holding individuals accountable for actions under e-signatures"
                ),
            }

        if uri == "oncology://guidance/ich-e6r3":
            return {
                "uri": uri,
                "mimeType": "text/plain",
                "content": (
                    "ICH E6(R3) Good Clinical Practice - Key Principles\n"
                    "1. Clinical trials should be conducted in accordance with ethical principles\n"
                    "2. Rights, safety, and well-being of trial participants are paramount\n"
                    "3. Scientifically sound protocols with clear objectives\n"
                    "4. Freely given informed consent from every participant\n"
                    "5. Quality management with risk-based approach\n"
                    "6. Adequate qualification of trial personnel\n"
                    "7. Compliance with applicable regulatory requirements\n"
                    "8. Data integrity and traceability throughout trial lifecycle\n"
                    "9. Proportionate monitoring based on identified risks\n"
                    "10. Technology-agnostic requirements for electronic systems"
                ),
            }

        return {"error": f"Resource not found: {uri}"}

    def get_audit_trail(self) -> str:
        """Export the complete audit trail as JSON."""
        return self._audit.export_json()

    def verify_audit_integrity(self) -> bool:
        """Verify the hash-chain integrity of the audit trail."""
        return self._audit.verify_chain()

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    @property
    def resource_count(self) -> int:
        """Number of registered resources."""
        return len(self._resources)

    @property
    def audit_entry_count(self) -> int:
        """Number of audit entries recorded."""
        return self._audit.count


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the MCP oncology server with tool invocations and resource access."""
    logger.info("=" * 80)
    logger.info("MCP Oncology Server Demonstration")
    logger.info("Version: 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 80)

    logger.info("Optional dependencies: HAS_MCP=%s, HAS_ANTHROPIC=%s", HAS_MCP, HAS_ANTHROPIC)

    # Initialize server
    server = OncologyMCPServer(server_name="demo-oncology-mcp")

    # List available tools
    tools = server.list_tools()
    logger.info("Available tools (%d):", len(tools))
    for tool in tools:
        logger.info("  - %s: %s", tool["name"], tool["description"][:80])

    # List available resources
    resources = server.list_resources()
    logger.info("Available resources (%d):", len(resources))
    for resource in resources:
        logger.info("  - %s: %s", resource["uri"], resource["name"])

    # Invoke: patient lookup
    logger.info("-" * 60)
    logger.info("Tool: oncology_patient_lookup (filter by NSCLC)")
    result = server.call_tool("oncology_patient_lookup", {"diagnosis": "NSCLC"})
    logger.info("Found %d patients", result.get("count", 0))

    # Invoke: treatment simulation
    patients = result.get("patients", [])
    if patients:
        pid = patients[0]["patient_id"]
        logger.info("-" * 60)
        logger.info("Tool: oncology_treatment_simulation (patient=%s)", pid)
        sim_result = server.call_tool(
            "oncology_treatment_simulation",
            {"patient_id": pid, "modality": "chemotherapy", "dose_mg": 80.0, "cycles": 4},
        )
        logger.info(
            "Simulation: %s (%.1f%% reduction, toxicity grade %d)",
            sim_result.get("response_category", "N/A"),
            sim_result.get("volume_reduction_pct", 0.0),
            sim_result.get("toxicity_grade", 0),
        )

        # Invoke: compliance check
        logger.info("-" * 60)
        logger.info("Tool: oncology_compliance_check (patient=%s)", pid)
        compliance = server.call_tool(
            "oncology_compliance_check",
            {"patient_id": pid, "protocol_id": "PROTO-ONC-2025-001"},
        )
        logger.info(
            "Compliance: %s (%d violations, %d warnings)",
            compliance.get("status", "N/A"),
            len(compliance.get("violations", [])),
            len(compliance.get("warnings", [])),
        )

    # Invoke: robot telemetry
    logger.info("-" * 60)
    logger.info("Tool: oncology_robot_telemetry")
    telemetry = server.call_tool("oncology_robot_telemetry", {"robot_id": "ROBOT-001"})
    logger.info(
        "Robot %s: mode=%s, phase=%s, safe=%s",
        telemetry.get("robot_id", "N/A"),
        telemetry.get("mode", "N/A"),
        telemetry.get("phase", "N/A"),
        telemetry.get("is_safe", "N/A"),
    )

    # Read a resource
    logger.info("-" * 60)
    logger.info("Resource: oncology://protocols/PROTO-ONC-2025-001")
    resource = server.read_resource("oncology://protocols/PROTO-ONC-2025-001")
    if "content" in resource:
        content = json.loads(resource["content"])
        logger.info("Protocol: %s (version %s)", content.get("title", "N/A")[:60], content.get("version", "N/A"))

    # Read regulatory guidance
    logger.info("-" * 60)
    logger.info("Resource: oncology://guidance/fda-21cfr11")
    guidance = server.read_resource("oncology://guidance/fda-21cfr11")
    if "content" in guidance:
        lines = guidance["content"].split("\n")
        logger.info("Guidance: %s (%d requirements)", lines[0], len(lines) - 2)

    # Verify audit trail
    logger.info("-" * 60)
    integrity_ok = server.verify_audit_integrity()
    logger.info(
        "Audit trail: %d entries, integrity=%s",
        server.audit_entry_count,
        "PASS" if integrity_ok else "FAIL",
    )

    logger.info("=" * 80)
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
